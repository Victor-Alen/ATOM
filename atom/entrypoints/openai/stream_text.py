"""Request-local incremental text decoding for fast tokenizers.

Token IDs remain untouched. Keep incomplete UTF-8 bytes buffered across
single-token and speculative multi-token chunks, and flush the final suffix.
"""
import codecs
import json
import weakref

_BYTE_TABLES = weakref.WeakKeyDictionary()


def _bytelevel_table(tokenizer, backend):
    """Use the exact reversible ByteLevel alphabet, only for a plain decoder."""
    if tokenizer in _BYTE_TABLES:
        return _BYTE_TABLES[tokenizer]
    table = None
    if backend.decoder is None:
        _BYTE_TABLES[tokenizer] = None
        return None
    state = json.loads(backend.decoder.__getstate__())
    if state.get("type") == "ByteLevel":
        byte_values = list(range(ord("!"), ord("~") + 1))
        byte_values += list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        alphabet = list(byte_values)
        missing = 0
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value)
                alphabet.append(256 + missing)
                missing += 1
        inverse = {chr(char): byte for byte, char in zip(byte_values, alphabet)}
        special_ids = {idx for idx, token in backend.get_added_tokens_decoder().items()
                       if token.special}
        candidate = {}
        for token, idx in backend.get_vocab().items():
            if idx in special_ids:
                candidate[idx] = b""
            elif all(char in inverse for char in token):
                candidate[idx] = bytes(inverse[char] for char in token)
            else:
                break
        else:
            table = candidate
    _BYTE_TABLES[tokenizer] = table
    return table


class StreamTextDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        self.stream = None
        self.token_ids = []
        self.emitted = ""
        self.byte_table = None
        self.utf8_decoder = None
        # Preserve existing behavior for wrappers whose post-decode cleanup can
        # rewrite already emitted text, or that lack the streaming decoder API.
        if self.backend is not None and not getattr(tokenizer, "clean_up_tokenization_spaces", False):
            self.byte_table = _bytelevel_table(tokenizer, self.backend)
            if self.byte_table is not None:
                self.utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")
            try:
                from tokenizers.decoders import DecodeStream
                self.stream = DecodeStream(skip_special_tokens=True)
            except ImportError:
                pass

    def decode(self, token_ids, *, finished=False):
        if self.stream is None and self.utf8_decoder is None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        self.token_ids.extend(token_ids)
        if self.utf8_decoder is not None:
            raw = b"".join(self.byte_table[token] for token in token_ids)
            chunk = self.utf8_decoder.decode(raw, final=finished)
        else:
            chunk = (self.stream.step(self.backend, token_ids) or "") if token_ids else ""
        self.emitted += chunk
        if finished:
            complete = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
            if not complete.startswith(self.emitted):
                raise RuntimeError("Incremental tokenizer output is not a prefix of final decoding")
            chunk += complete[len(self.emitted):]
            self.emitted = complete
        return chunk
