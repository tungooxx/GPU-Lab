"""Bounded Qwen receiver prefill for technical integration; no scientific release."""
from __future__ import annotations

def chunk_ranges(length: int, chunk_size: int = 256):
    if type(length) is not int or length < 1:
        raise ValueError("NONEMPTY_TOKEN_SEQUENCE_REQUIRED")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 512:
        raise ValueError("INVALID_PREFILL_CHUNK_SIZE")
    return [(start, min(start + chunk_size, length))
            for start in range(0, length, chunk_size)]

def forward_streamed(model, input_ids, *, restored_cache=None, chunk_size=256):
    """Consume each input token once with explicit absolute positions and full mask."""
    import torch
    if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise ValueError("BATCH_ONE_TOKENS_REQUIRED")
    ranges = chunk_ranges(int(input_ids.shape[-1]), chunk_size)
    cache = restored_cache
    initial = 0 if cache is None else int(cache.get_seq_length())
    logits = None
    for start, end in ranges:
        past = initial + start
        positions = torch.arange(past, initial + end, device=input_ids.device, dtype=torch.long)
        mask = torch.ones((1, initial + end), device=input_ids.device, dtype=torch.long)
        with torch.inference_mode():
            out = model(input_ids=input_ids[:, start:end],
                        attention_mask=mask, position_ids=positions.unsqueeze(0),
                        cache_position=positions, past_key_values=cache,
                        use_cache=True, return_dict=True)
        cache = out.past_key_values
        if int(cache.get_seq_length()) != initial + end:
            raise RuntimeError("CHUNK_CACHE_LENGTH_MISMATCH")
        logits = out.logits[:, -1, :].detach().cpu()
    return {"last_logits": logits, "cache": cache,
            "restored_cache_tokens": initial, "input_tokens": int(input_ids.shape[-1]),
            "chunks": len(ranges), "all_tokens_processed_exactly_once": True,
            "final_cache_tokens": int(cache.get_seq_length())}
