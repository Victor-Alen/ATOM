"""Request-local incremental text decoding for fast tokenizers.

Token IDs remain untouched. Keep incomplete UTF-8 bytes buffered across
single-token and speculative multi-token chunks, and flush the final suffix.
"""
class StreamTextDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        self.stream = None
        self.token_ids = []
        self.emitted = ""
        # Preserve existing behavior for wrappers whose post-decode cleanup can
        # rewrite already emitted text, or that lack the streaming decoder API.
        if self.backend is not None and not getattr(tokenizer, "clean_up_tokenization_spaces", False):
            try:
                from tokenizers.decoders import DecodeStream
                self.stream = DecodeStream(skip_special_tokens=True)
            except ImportError:
                pass

    def decode(self, token_ids, *, finished=False):
        if self.stream is None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        self.token_ids.extend(token_ids)
        chunk = (self.stream.step(self.backend, token_ids) or "") if token_ids else ""
        self.emitted += chunk
        if finished:
            complete = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
            if not complete.startswith(self.emitted):
                raise RuntimeError("Incremental tokenizer output is not a prefix of final decoding")
            chunk += complete[len(self.emitted):]
            self.emitted = complete
        return chunk
