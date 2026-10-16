"""Build the MuJoCo scene: Menagerie Panda + box lid + cylinder + obstacles + goal.

The scene is assembled with ``MjSpec`` so everything is parameterised from the
config. Obstacles and the goal marker are static world geoms whose positions
are edited at runtime, so an episode reset never recompiles the model.
"""
from __future__ import annotations

import mujoco
import numpy as np

from pushwm.config import ROOT

PANDA_XML = ROOT / "assets" / "franka_emika_panda" / "panda.xml"

G = mujoco.mjtGeom
PARK_Z = -1.0  # unused obstacle slots are moved here


def box_origin(cfg) -> np.ndarray:
    """World xy of the box-frame origin (inner floor corner with min x, min y)."""
    c = np.asarray(cfg.box.center, dtype=float)
    return c - 0.5 * np.array([cfg.box.size_x, cfg.box.size_y])


def floor_z(cfg) -> float:
    return float(cfg.box.floor_thickness)


def build_model(cfg) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.modelname = "capWM"
    spec.option.timestep = cfg.control.sim_timestep
    wb = spec.worldbody

    # Ground, light, cameras ------------------------------------------------
    wb.add_geom(name="ground", type=G.mjGEOM_PLANE, size=[0, 0, 0.05],
                rgba=[0.92, 0.92, 0.92, 1], contype=1, conaffinity=1)
    wb.add_light(name="sun", pos=[0.5, 0, 2.0], dir=[0, 0, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1], castshadow=True)
    spec.visual.global_.offwidth = 1280      # offscreen framebuffer for high-res renders
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient[:] = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse[:] = [0.4, 0.4, 0.4]

    cx, cy = cfg.box.center
    sx, sy = cfg.box.size_x, cfg.box.size_y
    cam_h = 1.25 * max(sx, sy) + 0.3
    # Top camera looks straight down (camera -z = world -z); image right = world +x.
    wb.add_camera(name="top", pos=[cx, cy, cam_h], quat=[1, 0, 0, 0], fovy=45)
    # Overhead camera framing the box for image-based policies (arm hidden at render time).
    wb.add_camera(name="overhead", pos=[cx, cy, 0.55], quat=[1, 0, 0, 0], fovy=45)
    # Isometric camera: whole robot and box in frame, fixed target between base and box.
    wb.add_body(name="iso_target", pos=[0.35, 0.0, 0.22])   # centres robot + box in the iso frame
    wb.add_camera(name="iso", pos=[cx + 0.66, cy - 0.84, 0.78], fovy=42,
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="iso_target")
    # Side camera shows the arm and the box together.
    wb.add_camera(name="side", pos=[cx + 0.9, cy - 0.9, 0.75], fovy=45,
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="cylinder")

    # Box lid: floor slab and four walls -------------------------------------
    ft, wh, wt = cfg.box.floor_thickness, cfg.box.wall_height, cfg.box.wall_thickness
    wb.add_geom(name="box_floor", type=G.mjGEOM_BOX,
                size=[sx / 2 + wt, sy / 2 + wt, ft / 2], pos=[cx, cy, ft / 2],
                rgba=[0.80, 0.78, 0.74, 1], friction=[cfg.box.floor_friction, 0.005, 0.0001])
    wall_rgba = [0.12, 0.16, 0.20, 1]
    zc = ft + wh / 2
    wb.add_geom(name="wall_xmin", type=G.mjGEOM_BOX, size=[wt / 2, sy / 2 + wt, wh / 2],
                pos=[cx - sx / 2 - wt / 2, cy, zc], rgba=wall_rgba)
    wb.add_geom(name="wall_xmax", type=G.mjGEOM_BOX, size=[wt / 2, sy / 2 + wt, wh / 2],
                pos=[cx + sx / 2 + wt / 2, cy, zc], rgba=wall_rgba)
    wb.add_geom(name="wall_ymin", type=G.mjGEOM_BOX, size=[sx / 2 + wt, wt / 2, wh / 2],
                pos=[cx, cy - sy / 2 - wt / 2, zc], rgba=wall_rgba)
    wb.add_geom(name="wall_ymax", type=G.mjGEOM_BOX, size=[sx / 2 + wt, wt / 2, wh / 2],
                pos=[cx, cy + sy / 2 + wt / 2, zc], rgba=wall_rgba)

    # Goal marker: flat, visual only ----------------------------------------
    wb.add_geom(name="goal", type=G.mjGEOM_CYLINDER, size=[cfg.task.goal_tol, 0.0005, 0],
                pos=[cx, cy, ft + 0.0006], rgba=[0.1, 0.75, 0.2, 1], contype=0, conaffinity=0)

    # Obstacles: static cylinders, parked until reset places them -----------
    for i in range(cfg.obstacles.max_slots):
        wb.add_geom(name=f"obstacle_{i}", type=G.mjGEOM_CYLINDER,
                    size=[cfg.obstacles.radius, cfg.obstacles.height / 2, 0],
                    pos=[cx, cy, PARK_Z], rgba=[0.85, 0.15, 0.1, 1])

    # Probe marker for kinematic replays of recorded clips: looks like the probe from
    # above, no collision, parked below the floor unless a replay places it.
    wb.add_geom(name="probe_marker", type=G.mjGEOM_CAPSULE, size=[cfg.probe.radius, 0.025, 0],
                pos=[cx, cy, PARK_Z], rgba=[0.95, 0.75, 0.6, 1], contype=0, conaffinity=0)

    # Pushed cylinder: free body -------------------------------------------
    r, h = cfg.cylinder.radius, cfg.cylinder.height
    cyl = wb.add_body(name="cylinder", pos=[cx, cy, ft + h / 2])
    cyl.add_freejoint(name="cylinder_free")
    cyl.add_geom(name="cylinder_geom", type=G.mjGEOM_CYLINDER, size=[r, h / 2, 0],
                 rgba=[0.15, 0.45, 0.9, 1], mass=cfg.cylinder.mass,
                 friction=[cfg.box.floor_friction, 0.005, 0.0001], condim=4)

    # End effector: finger-like probe between the open gripper fingers ------
    hand = spec.body("hand")
    if cfg.probe.enabled:
        pr = cfg.probe.radius
        half = 0.025
        zc_probe = cfg.probe.tip_offset - half - pr  # tip ends at local z = tip_offset
        hand.add_geom(name="probe", type=G.mjGEOM_CAPSULE, size=[pr, half, 0],
                      pos=[0, 0, zc_probe], rgba=[0.95, 0.75, 0.6, 1],
                      friction=[cfg.box.floor_friction, 0.005, 0.0001])
        tcp_z = zc_probe + half + pr
    else:
        tcp_z = 0.117  # closed fingertip pads
    hand.add_site(name="tcp", pos=[0, 0, tcp_z], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.0])

    # Arm actuators: keep Menagerie's stiffness but lower the velocity damping so
    # the 87 Nm torque limit is not spent fighting kv at pushing speeds.
    for act in spec.actuators[:7]:
        act.biasprm[2] = -cfg.control.kv
    # Gravity compensation on every arm link: position servos then hold pose
    # without sag, as the real Panda's controller does.
    body = spec.body("link0")
    todo = [body]
    while todo:
        b = todo.pop()
        b.gravcomp = 1.0
        todo.extend(b.bodies)

    model = spec.compile()
    return model


def obstacle_geom_ids(model: mujoco.MjModel, cfg) -> list[int]:
    return [model.geom(f"obstacle_{i}").id for i in range(cfg.obstacles.max_slots)]
