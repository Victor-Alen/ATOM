#!/usr/bin/env python3
"""Make the teacher-forced gate incremental and resumable for long runs.

Writes the summary after every prompt (so a 200-prompt run survives a restart)
and skips prompts already present in the output file.
"""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/teacher_forced_gate.py")
s = p.read_text()

old_records = """    records = []
    with open(args.reference, encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= args.limit:
                break
            records.append(json.loads(line))

    results = []"""
new_records = """    records = []
    with open(args.reference, encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= args.limit:
                break
            records.append(json.loads(line))

    results = []
    done_ids = set()
    out_path = Path(args.output)
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("results", [])
            done_ids = {r["id"] for r in results}
            print(f"resuming: {len(done_ids)} prompts already done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not resume ({exc}); starting fresh", flush=True)
            results = []
            done_ids = set()

    def save():
        conf_n = sum(e["buckets"]["gap_ge_1.0"]["n"] for e in results)
        conf_m = sum(e["buckets"]["gap_ge_1.0"]["matched"] for e in results)
        tie_n = sum(e["buckets"]["gap_lt_1.0"]["n"] for e in results)
        tie_m = sum(e["buckets"]["gap_lt_1.0"]["matched"] for e in results)
        matched_all = sum(e["matched_all"] for e in results)
        verified_all = sum(e["verified"] for e in results)
        summary = {
            "variants": str(args.url),
            "prompts": len(results),
            "positions_verified": verified_all,
            "positions_matched": matched_all,
            "all_rate": matched_all / verified_all if verified_all else 0.0,
            "confident_positions": conf_n,
            "confident_matched": conf_m,
            "confident_rate": conf_m / conf_n if conf_n else 0.0,
            "near_tie_positions": tie_n,
            "near_tie_matched": tie_m,
            "near_tie_rate": tie_m / tie_n if tie_n else 0.0,
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary))
        return summary"""
assert old_records in s
s = s.replace(old_records, new_records, 1)

old_loop = """    for rec in records:
        pid = rec["id"]"""
new_loop = """    for rec in records:
        pid = rec["id"]
        if pid in done_ids:
            continue"""
assert old_loop in s
s = s.replace(old_loop, new_loop, 1)

old_after = """        results.append(entry)
        print("""
new_after = """        results.append(entry)
        summary = save()
        print("""
assert old_after in s
s = s.replace(old_after, new_after, 1)

# replace the tail summary block with the save() version
old_tail = s[s.index("    conf_n = sum(e[\"buckets\"][\"gap_ge_1.0\"][\"n\"] for e in results)"):]
new_tail = """    summary = save()
    conf_n = summary["confident_positions"]
    conf_m = summary["confident_matched"]
    tie_n = summary["near_tie_positions"]
    tie_m = summary["near_tie_matched"]
    print(
        f"CONFIDENT (HF gap >= 1.0): {conf_m}/{conf_n} = {(conf_m/conf_n*100 if conf_n else 0):.2f}%  |  "
        f"NEAR-TIE (gap < 1.0): {tie_m}/{tie_n} = {(tie_m/tie_n*100 if tie_n else 0):.2f}%  |  "
        f"ALL: {summary['positions_matched']}/{summary['positions_verified']} = {summary['all_rate']*100:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
"""
s = s.replace(old_tail, new_tail, 1)

p.write_text(s)
import ast
ast.parse(s)
print("incremental + resumable patch applied")
