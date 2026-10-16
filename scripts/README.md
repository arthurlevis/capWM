# scripts

One command per step, in pipeline order. All paths are relative to the repo root; set `MUJOCO_GL=egl` for headless rendering.

| step | script | writes |
|---|---|---|
| calibrate the box once per session | `calibrate_all.py` (auto, with QA sheet) or `calibrate_box.py --click` (manual corners) | `data/calib/` |
| check every clip | `check_clips.py` | `data/processed/clip_check.json` |
| videos → trajectories | `extract_videos.py` | `data/processed/video/`, `data/processed/tracks/` |
| train the world model | `train_state_wm.py` | a checkpoint (`models/`) |
| world-model figures | `plot_training.py`, `plot_model_insights.py` | `docs/figures/`, `docs/results/*_summary.json` |
| simulator pushes for the sim-data model and the transfer check | `collect_sim_data.py`, `eval_transfer.py` | `data/processed/sim/`, `outputs/eval/` |
| run a planner on a benchmark | `plan.py --policy wm|rule --opposite|--hard` | `outputs/eval/`, `outputs/logs/` |
| retarget clips into robot demonstrations (SmolVLA) | `retarget_demos.py` | `data/demos_retargeted/` (ignored) |
| evaluate an image policy closed-loop | `eval_policy_sim.py` | `outputs/eval/`, `outputs/logs/` |
| collect results and build the table | `results_table.py` | `docs/results/` |
| benchmark figure | `plot_benchmark.py` | `docs/figures/benchmark.png` |
| render episodes | `render_all_episodes.py` (schematic gif per episode), `render_schematic.py` (several episodes side by side), `render_panda_episodes.py` (isometric robot view) | `docs/media/` |
| tracking gifs | `make_media.py` | `docs/media/` |
