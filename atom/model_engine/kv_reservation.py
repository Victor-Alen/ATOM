"""Bounded-generation KV admission for exact, non-recomputing decode."""
def required_blocks(seq, block_size, mtp_k):
    # One deferred forward may already be in flight when max_tokens is seen;
    # can_append additionally checks the next verification's lookahead.
    tokens = seq.num_prompt_tokens + seq.max_tokens + 2 * (mtp_k + 1)
    return (tokens + block_size - 1) // block_size


def can_admit(seq, running, block_manager, mtp_k):
    block_size = block_manager.block_size
    reserved = sum(max(0, required_blocks(s, block_size, mtp_k) - len(s.block_table))
                   for s in running)
    needed = required_blocks(seq, block_size, mtp_k)
    return needed + reserved <= len(block_manager.free_block_ids_set)
