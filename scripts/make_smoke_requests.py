import json
from pathlib import Path

from transformers import AutoTokenizer

model = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model)

for label in ("a", "c"):
    dataset = Path(f"/data/datasets/scenario_{label}_prompts.jsonl")
    item = json.loads(dataset.open(encoding="utf-8").readline())
    prompt = item["prompt"]
    token_count = len(tokenizer(prompt)["input_ids"])
    request = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    output = Path(f"/workspace/smoke/request_{label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(request), encoding="utf-8")
    print(label, token_count, output)
