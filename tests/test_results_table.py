import json
import subprocess
import sys

from tests.conftest import ROOT


def test_results_table_from_a_small_eval(tmp_path):
    evals = tmp_path / "eval"; evals.mkdir()
    res = [dict(success=True, collided=False, timeout=False, steps=20, min_obstacle_clearance=0.03),
           dict(success=False, collided=True, timeout=False, steps=5, min_obstacle_clearance=-0.001)]
    (evals / "eval_barrier_world_planner.json").write_text(json.dumps(dict(results=res)))
    out = tmp_path / "docs"
    subprocess.run([sys.executable, str(ROOT / "scripts/results_table.py"), "--evals", str(evals), "--out", str(out)], check=True, capture_output=True)
    rows = json.loads((out / "results_table.json").read_text())
    assert len(rows) == 1
    r = rows[0]
    assert (r["layout"], r["controller"], r["n"], r["success"], r["collision"], r["timeout"]) == ("barrier", "world_planner", 2, 1, 1, 0)
    assert r["steps_median"] == 20.0 and abs(r["clearance_mm"] - 14.5) < 1e-6
    assert (out / "eval_barrier_world_planner.json").exists()
