import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)


def gen(text, steps=64):
    p = {"model": MODEL, "prompt": text, "max_tokens": steps, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    r = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())["token_ids"]


def run(label, path, limit=40, maxlen=12000):
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if len(r["prompt_token_ids"]) <= maxlen:
            recs.append(r)
        if len(recs) >= limit:
            break
    tot, ex = 0.0, 0
    for rec in recs:
        ref = rec["token_ids"][:64]
        got = gen(tok.decode(rec["prompt_token_ids"]))
        n = min(len(got), len(ref))
        a = sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0.0
        tot += a
        ex += (a == 1.0)
        print(f"{rec['id']}: {a:.3f} {'EXACT' if a == 1.0 else ''}", flush=True)
    print(f"{label}: agreement={tot/len(recs):.4f} exact={ex}/{len(recs)}", flush=True)


run("B", "/workspace/correctness/hf_scenario_b_20_eager.jsonl", limit=20)
run("C", "/workspace/correctness/hf_scenario_c_200_eager.jsonl", limit=40)
print("### BC_DONE", flush=True)
