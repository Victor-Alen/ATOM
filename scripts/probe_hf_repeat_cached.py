#!/usr/bin/env python3
"""Repeat native HF Mellum greedy generation under the saved eager/cache protocol."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/ATOM-tritonmoe/scripts")
from hf_reference_generate import MODEL_PATH, load_checkpoint_experts, load_mellum_classes

OUT = Path("/workspace/logs/step0_continuation/hf_repeat_cached.json")
SOURCES = (
    ("A-0000", "/workspace/correctness/hf_scenario_a_20_eager.jsonl", 16),
    ("C-0001", "/workspace/correctness/hf_scenario_c_first5_eager.jsonl", 8),
)


def record(path, item_id):
    with open(path, encoding="utf-8") as handle:
        return next(json.loads(line) for line in handle if json.loads(line)["id"] == item_id)


def main():
    config_cls, model_cls = load_mellum_classes()
    config = config_cls.from_pretrained(MODEL_PATH)
    model = model_cls.from_pretrained(
        MODEL_PATH, config=config, dtype=torch.bfloat16, attn_implementation="eager"
    )
    load_checkpoint_experts(model, MODEL_PATH)
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    results = []
    for item_id, path, count in SOURCES:
        saved = record(path, item_id)
        inputs = tokenizer(saved["prompt"], return_tensors="pt").to(model.device)
        assert inputs.input_ids[0].tolist() == saved["prompt_token_ids"]
        streams = []
        for _ in range(3):
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=False, num_beams=1,
                    max_new_tokens=count, min_new_tokens=count, use_cache=True,
                )
            streams.append(output[0, inputs.input_ids.shape[1]:].tolist())
        result = {
            "id": item_id, "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": count, "attention": "eager",
            "use_cache": True, "dtype": "bf16",
            "three_runs_identical": all(s == streams[0] for s in streams),
            "matches_saved_prefix": streams[0] == saved["token_ids"][:count],
            "streams": streams, "saved_prefix": saved["token_ids"][:count],
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    main()
