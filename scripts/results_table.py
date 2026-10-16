"""Collect the evaluation files the README cites from outputs/ into docs/results/
and build the results table (markdown + json). This is the only step that writes
into docs/results.

    python scripts/results_table.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

LAYOUTS = ["opposite", "barrier"]
MODELS = [("world_planner", "world-model planner (phone data)"), ("sim_planner", "planner with sim-data model (upper bound)"),
          ("rule_planner", "planner with hand-written rigid-contact model"), ("smolvla", "SmolVLA fine-tuned on retargeted clips")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", default="outputs/eval", help="where scripts/plan.py and eval_policy_sim.py wrote their json")
    p.add_argument("--out", default="docs/results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.evals).glob("eval_*.json")) + sorted(Path(a.evals).glob("transfer_*.json")):
        shutil.copy2(f, out / f.name)
    rows, md = [], ["| layout | controller | success | collision | timeout | median steps (success) | median min clearance |", "|---|---|---:|---:|---:|---:|---:|"]
    for L in LAYOUTS:
        for key, label in MODELS:
            f = Path(a.evals) / f"eval_{L}_{key}.json"
            if not f.exists():
                continue
            r = json.load(open(f))["results"]
            n = len(r); s = sum(x["success"] for x in r); c = sum(x["collided"] for x in r); t = n - s - c
            st = [x["steps"] for x in r if x["success"]]
            cl = np.array([x["min_obstacle_clearance"] for x in r if x["min_obstacle_clearance"] is not None])
            row = dict(layout=L, controller=key, n=n, success=s, collision=c, timeout=t,
                       steps_median=float(np.median(st)) if st else None, clearance_mm=float(np.median(cl) * 1000))
            rows.append(row)
            md.append(f"| {L} | {label} | {s}/{n} | {c} | {t} | {row['steps_median'] if st else '–'} | {row['clearance_mm']:.0f} mm |")
    (out / "results_table.json").write_text(json.dumps(rows, indent=1))
    (out / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
