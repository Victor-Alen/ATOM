"""Audit saved EvalScope requests and summarize per-run A/B/C metrics."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def summarize(root, repeats, scenarios):
    protocol = json.loads((root / "protocol.json").read_text())
    records = []
    for repeat in range(repeats):
        for scene in scenarios:
            folder = root / f"r{repeat}" / scene
            expected = protocol["counts"][scene]
            files = {}
            for name in ("benchmark_summary.json", "benchmark_percentile.json", "benchmark_data.db"):
                found = list(folder.rglob(name))
                if len(found) != 1:
                    raise ValueError(f"Expected one {name} in {folder}; found {len(found)}")
                files[name] = found[0]
            if (folder / "exit_code").read_text().strip() != "0":
                raise ValueError(f"Client failed: {folder}")
            summary = json.loads(files["benchmark_summary.json"].read_text())
            percentile = json.loads(files["benchmark_percentile.json"].read_text())
            if summary["Success Requests"] != expected or summary["Failed Requests"] != 0:
                raise ValueError(f"Incomplete or failed requests: {folder}")
            with sqlite3.connect(files["benchmark_data.db"].resolve().as_uri() + "?mode=ro", uri=True) as db:
                count = db.execute("SELECT count(*) FROM result").fetchone()[0]
                success = db.execute("SELECT success,count(*) FROM result GROUP BY success").fetchall()
                tokens = db.execute("SELECT completion_tokens,count(*) FROM result GROUP BY completion_tokens").fetchall()
                streamed = db.execute("SELECT is_stream,count(*) FROM result GROUP BY is_stream").fetchall()
            if count != expected or success != [(1, expected)] or tokens != [(256, expected)] or streamed != [(1, expected)]:
                raise ValueError(f"Request audit failed: {folder}; rows={count}, success={success}, tokens={tokens}, stream={streamed}")
            command = json.loads((folder / "command.json").read_text())
            if "--stream" not in command:
                raise ValueError(f"Missing stream flag: {folder}")
            extra = json.loads(command[command.index("--extra-args") + 1])
            if extra.get("ignore_eos") is not True or extra.get("temperature") != 0:
                raise ValueError(f"Unexpected generation protocol: {folder}")
            def pct(p, key):
                return next(row[key] for row in percentile if row["Percentiles"] == p)
            records.append({"repeat": repeat, "scenario": scene, "requests": count,
                            "output_tokens_per_s_gpu": summary["Output Throughput (tok/s)"] / protocol["num_gpus"],
                            "TPOT_p50_ms": pct("50%", "TPOT (ms)"),
                            "TPOT_p99_ms": pct("99%", "TPOT (ms)"),
                            "TTFT_p99_ms": pct("99%", "TTFT (ms)")})
    metrics = {}
    for scene, key in (("A", "output_tokens_per_s_gpu"), ("B", "TPOT_p50_ms"), ("C", "TPOT_p99_ms"), ("C", "TTFT_p99_ms")):
        if scene in scenarios:
            values = [row[key] for row in records if row["scenario"] == scene]
            metrics[f"{scene}_{key}"] = {"rounds": values, "median": statistics.median(values)}
    return {"scope": protocol["scope"], "num_gpus": protocol["num_gpus"], "mtp": protocol["mtp"],
            "repeats": repeats, "a_source": protocol["a_source"], "all_requests_successful": True,
            "all_outputs_256": True, "all_requests_streamed": True, "metrics": metrics, "records": records,
            "interpretation": "Per-round percentiles followed by median across rounds; screening is not full acceptance. Correctness is validated separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--scenarios", default="ABC")
    args = parser.parse_args()
    if args.expected_repeats < 1 or not args.scenarios or len(set(args.scenarios)) != len(args.scenarios) or any(s not in "ABC" for s in args.scenarios):
        parser.error("Specify positive repeats and distinct A/B/C scenario letters")
    result = summarize(args.run_dir, args.expected_repeats, args.scenarios)
    output = args.run_dir / "audited_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "scope": result["scope"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
