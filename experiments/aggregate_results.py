"""
Aggregate Phase 5 results.

Reads the per-episode JSONL emitted by run_experiments.py and produces a
summary table (one row per variant × condition) with the proposal's
headline metrics:

    n            — number of episodes
    success      — Success Rate (proposal: SR)
    collision    — Collision Rate (proposal: CR)
    timeout      — Timeout rate
    per_mean     — mean Path Efficiency Ratio over successful episodes
    steps_mean   — mean steps per episode (all outcomes)
    replans_mean — mean Replanning Count (proposal: MRC)
    heals_mean   — mean Healer invocations per episode
    llm_calls    — mean LLM calls per episode
    tokens_mean  — mean LLM tokens per episode
    wall_ms_mean — mean wall-clock time per episode

Writes a CSV next to the inputs and prints a Markdown table.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_one(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    success    = sum(1 for r in records if r["outcome"] == "success")
    collision  = sum(1 for r in records if r["outcome"] == "collision")
    timeout    = sum(1 for r in records if r["outcome"] == "timeout")

    pers       = [r["per"] for r in records if r["outcome"] == "success" and r.get("per") is not None]
    return {
        "n":             n,
        "success_rate":  success / n,
        "collision_rate": collision / n,
        "timeout_rate":  timeout / n,
        "per_mean":      _safe_mean(pers),
        "steps_mean":    _safe_mean([r["steps_taken"] for r in records]),
        "replans_mean":  _safe_mean([r["replan_count"] for r in records]),
        "heals_mean":    _safe_mean([r.get("heal_count", 0) for r in records]),
        "llm_calls_mean": _safe_mean([r.get("llm_call_count", 0) for r in records]),
        "tokens_mean":   _safe_mean([r.get("llm_total_tokens", 0) for r in records]),
        "wall_ms_mean":  _safe_mean([r.get("wall_clock_ms", 0.0) for r in records]),
    }


def aggregate(input_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl") or fname.startswith("_memstore__"):
            continue
        path    = os.path.join(input_dir, fname)
        records = _load_jsonl(path)
        if not records:
            continue
        variant   = records[0].get("variant",   fname.split("__")[0])
        condition = records[0].get("condition", fname.split("__")[1].replace(".jsonl", ""))
        rows.append({
            "variant":   variant,
            "condition": condition,
            **_aggregate_one(records),
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    cols = [
        ("variant",          "{}"),
        ("condition",        "{}"),
        ("n",                "{}"),
        ("success_rate",     "{:.2f}"),
        ("collision_rate",   "{:.2f}"),
        ("timeout_rate",     "{:.2f}"),
        ("per_mean",         "{:.2f}"),
        ("steps_mean",       "{:.1f}"),
        ("replans_mean",     "{:.2f}"),
        ("heals_mean",       "{:.2f}"),
        ("llm_calls_mean",   "{:.1f}"),
        ("tokens_mean",      "{:.0f}"),
        ("wall_ms_mean",     "{:.0f}"),
    ]
    headers = [c[0] for c in cols]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    # Order: variants × conditions in a stable layout
    order_v = ["astar", "dstar_lite", "agent_noreflection", "agent_full"]
    order_c = ["static", "dynamic_low", "dynamic_high"]
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            order_v.index(r["variant"])   if r["variant"]   in order_v else 99,
            order_c.index(r["condition"]) if r["condition"] in order_c else 99,
        ),
    )
    for r in rows_sorted:
        cells = [fmt.format(r.get(name, "")) for name, fmt in cols]
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Phase 5 JSONL results.")
    ap.add_argument("--input",  default="results/logs/pilot",
                    help="directory containing per-variant JSONL files")
    ap.add_argument("--csv",    default=None,
                    help="output CSV path (default: <input>/summary.csv)")
    args = ap.parse_args()

    rows = aggregate(args.input)
    if not rows:
        print(f"No JSONL records found in {args.input!r}.")
        return

    csv_path = args.csv or os.path.join(args.input, "summary.csv")
    write_csv(rows, csv_path)
    print_markdown(rows)
    print(f"\n→ {csv_path}")


if __name__ == "__main__":
    main()
