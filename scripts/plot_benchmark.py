"""Stacked bars of success / collision / timeout per controller and layout, from
docs/results/results_table.json.

    python scripts/plot_benchmark.py --out docs/figures/benchmark.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"world_planner": "world", "sim_planner": "sim", "rule_planner": "b0", "smolvla": "smolvla"}
ORDER = ["rule_planner", "world_planner", "sim_planner", "smolvla"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="docs/results/results_table.json")
    p.add_argument("--out", default="docs/figures/benchmark.png")
    a = p.parse_args()
    rows = json.load(open(a.table))
    layouts = ["barrier", "opposite"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, L in zip(axes, layouts):
        rs = {r["controller"]: r for r in rows if r["layout"] == L}
        keys = [k for k in ORDER if k in rs]
        x = np.arange(len(keys))
        s = np.array([rs[k]["success"] for k in keys]); c = np.array([rs[k]["collision"] for k in keys]); t = np.array([rs[k]["timeout"] for k in keys])
        ax.bar(x, s, color="#2a9d8f", label="success"); ax.bar(x, c, bottom=s, color="#e76f51", label="collision"); ax.bar(x, t, bottom=s + c, color="#bdbdbd", label="timeout")
        for xi, v in zip(x, s):
            ax.text(xi, v + 1, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([LABELS[k] for k in keys], fontsize=11); ax.set_title(f"{L.capitalize()} layout, 100 episodes")
        ax.set_ylim(0, 105); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=1, fontsize=9, frameon=False, bbox_to_anchor=(0.995, 0.93))
    fig.suptitle("Push to goal around obstacles: failure on obstacle contact or > 8 s", fontsize=11, x=0.42)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95)); fig.savefig(a.out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", a.out)


if __name__ == "__main__":
    main()
