"""Pushing environment: random start, goal and obstacles; Cartesian control of a
finger-like probe on a Franka Panda; state logged in the box frame in metres.

The planner only ever sees ``obs``: probe tip xy, cylinder xy, goal xy and the
obstacle list, all in box coordinates. The action is a tip displacement (dx, dy)
per control step, which is the same quantity we extract from the phone videos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from pushwm.config import load_config
from pushwm.sim.scene import PARK_Z, build_model, box_origin, floor_z, obstacle_geom_ids


@dataclass
class Obs:
    finger: np.ndarray            # (2,) probe tip xy, box frame
    cylinder: np.ndarray          # (2,) cylinder centre xy, box frame
    goal: np.ndarray              # (2,) goal xy, box frame
    obstacles: np.ndarray         # (n, 3) x, y, radius
    cylinder_yaw: float = 0.0
    contact: bool = False
    collision: bool = False       # cylinder or probe touched an obstacle (episode failure)
    success: bool = False
    t: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(finger=self.finger.copy(), cylinder=self.cylinder.copy(), goal=self.goal.copy(),
                    obstacles=self.obstacles.copy(), contact=self.contact, collision=self.collision,
                    success=self.success, t=self.t)


class PushEnv:
    def __init__(self, cfg=None, seed: int | None = None):
        self.cfg = cfg or load_config()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._renderers: dict[str, mujoco.Renderer] = {}

        m = self.model
        self.tcp_site = m.site("tcp").id
        self.cyl_body = m.body("cylinder").id
        self.cyl_qpos = m.jnt_qposadr[m.joint("cylinder_free").id]
        self.cyl_dof = m.jnt_dofadr[m.joint("cylinder_free").id]
        self.goal_geom = m.geom("goal").id
        self.obs_geoms = obstacle_geom_ids(m, self.cfg)
        self.arm_dofs = np.arange(7)
        self.home_qpos = m.key("home").qpos.copy()
        self.origin = box_origin(self.cfg)
        self.size = np.array([self.cfg.box.size_x, self.cfg.box.size_y])
        self.tip_z = floor_z(self.cfg) + self.cfg.probe.tip_height

        # Desired tool orientation: pointing straight down, taken from the home pose.
        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)
        self.R_des = self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()
        self.q_des = np.zeros(4)
        mujoco.mju_mat2Quat(self.q_des, self.R_des.ravel())

        self.target = np.zeros(2)   # commanded tip xy in box frame
        self.q_cmd = self.home_qpos[:7].copy()
        self._collided = False
        self._active_obs_geoms = set()
        hand_bodies = (m.body("left_finger").id, m.body("right_finger").id, m.body("hand").id)
        self._collider_geoms = {m.geom("cylinder_geom").id} | ({m.geom("probe").id} if self.cfg.probe.enabled else set()) \
            | {g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies}
        self.obstacles = np.zeros((0, 3))
        self.goal = np.zeros(2)
        self.t = 0

    # ------------------------------------------------------------------ frames
    def to_world(self, xy_box):
        return np.asarray(xy_box, dtype=float) + self.origin

    def to_box(self, xy_world):
        return np.asarray(xy_world, dtype=float)[..., :2] - self.origin

    # ------------------------------------------------------------------ reset
    def _sample_free_point(self, margin, avoid, min_dist):
        for _ in range(500):
            p = self.rng.uniform(margin, self.size - margin)
            if all(np.linalg.norm(p - a) >= d for a, d in zip(avoid, min_dist)):
                return p
        raise RuntimeError("could not sample a free point")

    def _opposite_sides_layout(self):
        """Chessboard layout: files a-h across x (width), ranks 1-8 along y (length).
        Start in one two-file (or two-rank) edge zone, goal in the opposite one, random
        axis and direction; 4-6 obstacles at random in the band between the zones."""
        cfg = self.cfg
        r_cyl, r_ob = cfg.cylinder.radius, cfg.obstacles.radius
        margin = r_cyl + 0.01                     # keep the cap off the walls
        axis = int(self.rng.integers(2))          # 0: along width (x), 1: along length (y)
        L = self.size[axis]
        zone = 2 * L / 8                          # two files or two ranks
        lo_zone = (margin, zone)
        hi_zone = (L - zone, L - margin)
        if self.rng.random() < 0.5:
            lo_zone, hi_zone = hi_zone, lo_zone
        other = 1 - axis

        def pt(z):
            p = np.zeros(2)
            p[axis] = self.rng.uniform(*z)
            p[other] = self.rng.uniform(margin, self.size[other] - margin)
            return p
        start, goal = pt(lo_zone), pt(hi_zone)
        n = int(self.rng.integers(4, 7))
        keep_end = r_cyl + r_ob + cfg.task.obstacle_clearance + 0.01
        band = (zone + r_ob + 0.005, L - zone - r_ob - 0.005)
        placed = []
        for _ in range(2000):
            if len(placed) == n:
                break
            q = np.zeros(2)
            q[axis] = self.rng.uniform(*band)
            q[other] = self.rng.uniform(r_ob + 0.01, self.size[other] - r_ob - 0.01)
            if np.linalg.norm(q - start) < keep_end or np.linalg.norm(q - goal) < keep_end:
                continue
            if any(np.linalg.norm(q - p) < 2 * r_ob + 0.03 for p in placed):
                continue
            placed.append(q)
        obstacles = np.array([[p[0], p[1], r_ob] for p in placed]).reshape(-1, 3)
        return start, goal, obstacles

    def reset(self, start=None, goal=None, obstacles=None, n_obstacles=None, obstacle_on_line=False,
              opposite_sides=False) -> Obs:
        """``obstacle_on_line``: a three-obstacle barrier across the start-goal segment.
        ``opposite_sides``: chessboard edge-zone layout with 4-6 obstacles in between."""
        cfg, m, d = self.cfg, self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        r_cyl = cfg.cylinder.radius
        wm = cfg.task.wall_margin
        if opposite_sides:
            start, goal, obstacles = self._opposite_sides_layout()
        min_sg = max(cfg.task.min_start_goal_dist, 0.12) if obstacle_on_line else cfg.task.min_start_goal_dist

        fixed_start = np.asarray(start, float) if start is not None else None
        for _ in range(50):   # sample start and goal jointly; a corner start may have no far goal
            start = fixed_start if fixed_start is not None else self._sample_free_point(wm, [], [])
            if goal is not None:
                break
            try:
                goal = self._sample_free_point(wm, [start], [min_sg])
                break
            except RuntimeError:
                if fixed_start is not None:
                    raise
        self.goal = np.asarray(goal, float)

        if obstacles is None:
            n = n_obstacles if n_obstacles is not None else int(self.rng.integers(cfg.obstacles.n_min, cfg.obstacles.n_max + 1))
            n = min(n, cfg.obstacles.max_slots)
            placed = []
            keep = r_cyl + cfg.obstacles.radius + cfg.task.obstacle_clearance + 0.01
            if obstacle_on_line and n > 0:
                # A barrier of three obstacles across the start-goal line, spaced so the
                # cylinder cannot pass between them; a straight push is blocked.
                seg = self.goal - start
                perp = np.array([-seg[1], seg[0]]) / max(np.linalg.norm(seg), 1e-6)
                mid = start + self.rng.uniform(0.45, 0.55) * seg
                gap = 2 * cfg.obstacles.radius + 0.006
                n = max(n, 3)
                for k in (-1, 0, 1):
                    q = mid + perp * k * gap
                    if np.all(q > cfg.obstacles.radius + 0.01) and np.all(q < self.size - cfg.obstacles.radius - 0.01):
                        placed.append(q)
            for _ in range(n - len(placed)):
                avoid = [start, self.goal] + [p for p in placed]
                dist = [keep, keep] + [2 * cfg.obstacles.radius + 0.03] * len(placed)
                placed.append(self._sample_free_point(cfg.obstacles.radius + 0.01, avoid, dist))
            obstacles = np.array([[p[0], p[1], cfg.obstacles.radius] for p in placed]).reshape(-1, 3)
        self.obstacles = np.asarray(obstacles, float).reshape(-1, 3)

        # Place static geoms (goal marker, obstacles) by editing the model.
        ft = floor_z(cfg)
        m.geom_pos[self.goal_geom, :2] = self.to_world(self.goal)
        for i, gid in enumerate(self.obs_geoms):
            if i < len(self.obstacles):
                m.geom_pos[gid] = [*self.to_world(self.obstacles[i, :2]), ft + cfg.obstacles.height / 2]
                m.geom_size[gid, 0] = self.obstacles[i, 2]
            else:
                m.geom_pos[gid, 2] = PARK_Z

        # Cylinder pose.
        q = self.cyl_qpos
        d.qpos[q:q + 3] = [*self.to_world(start), ft + cfg.cylinder.height / 2]
        d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        d.qvel[self.cyl_dof:self.cyl_dof + 6] = 0

        self._active_obs_geoms = set(self.obs_geoms[: len(self.obstacles)])
        # Start the probe at a random free spot away from the cylinder and the obstacles, then settle.
        avoid = [start] + [o[:2] for o in self.obstacles]
        dist = [r_cyl + cfg.probe.radius + 0.04] + [o[2] + cfg.probe.radius + 0.02 for o in self.obstacles]
        tip0 = self._sample_free_point(0.03, avoid, dist)
        self.target = tip0.copy()
        self.q_cmd = d.qpos[:7].copy()
        d.ctrl[:7] = self.q_cmd
        d.ctrl[7] = 0  # gripper closed around the probe (actuator8: 0..255 -> 0..0.04 m)
        mujoco.mj_forward(m, d)
        self._settle(self.target)
        # Re-seat the cylinder in case the arm's descent disturbed anything.
        d.qpos[q:q + 3] = [*self.to_world(start), ft + cfg.cylinder.height / 2]
        d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        d.qvel[self.cyl_dof:self.cyl_dof + 6] = 0
        mujoco.mj_forward(m, d)
        self._collided = False
        self.t = 0
        return self.observe()

    # ------------------------------------------------------------------ control
    def _ik_update(self, tip_world_xy, pos_step=0.006, rot_step=0.06):
        """Kinematic damped-least-squares IK on the commanded joint vector.

        The Jacobian is evaluated at the *commanded* configuration (a separate
        MjData), so the command is a smooth kinematic trajectory and the PD
        actuators only have to track it. Returns the remaining tip error.
        """
        m, cfg = self.model, self.cfg
        ik = self.ik_data
        ik.qpos[:7] = self.q_cmd
        mujoco.mj_kinematics(m, ik)
        mujoco.mj_comPos(m, ik)
        pos_des = np.array([tip_world_xy[0], tip_world_xy[1], self.tip_z])
        e_pos = pos_des - ik.site_xpos[self.tcp_site]
        err = float(np.linalg.norm(e_pos))
        if err > pos_step:
            e_pos *= pos_step / err
        R_cur = ik.site_xmat[self.tcp_site].reshape(3, 3)
        R_err = self.R_des @ R_cur.T                      # world-frame rotation taking current to desired
        q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, R_err.ravel())
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)
        n_rot = np.linalg.norm(e_rot)
        if n_rot > rot_step:
            e_rot *= rot_step / n_rot
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, ik, jacp, jacr, self.tcp_site)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        e = np.concatenate([e_pos, cfg.control.ik_rot_gain * e_rot])
        lam = cfg.control.ik_damping
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
        N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
        dq += N @ (0.05 * (self.home_qpos[:7] - self.q_cmd))
        lo, hi = m.jnt_range[:7, 0] + 0.02, m.jnt_range[:7, 1] - 0.02
        self.q_cmd = np.clip(self.q_cmd + dq, lo, hi)
        # Anti-windup: if the arm is physically blocked (wall, obstacle, jammed
        # cylinder) the command must not run away from the measured joints.
        q = self.data.qpos[:7]
        self.q_cmd = np.clip(self.q_cmd, q - cfg.control.windup_limit, q + cfg.control.windup_limit)
        self.data.ctrl[:7] = self.q_cmd
        return err

    def _check_collision(self):
        """True if the cap or the hand/probe touches an active obstacle right now.
        Uses MuJoCo contacts and a geometric overlap test (a light cap squeezed
        between probe and obstacle can pop through within a few milliseconds)."""
        m, d = self.model, self.data
        n_act = len(self.obstacles)
        if n_act == 0:
            return False
        active = self._active_obs_geoms
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in active or g2 in active) and (g1 in self._collider_geoms or g2 in self._collider_geoms):
                return True
        cyl_xy = d.xpos[self.cyl_body][:2]
        tip_xy = d.site_xpos[self.tcp_site][:2]
        for gid, ob in zip(self.obs_geoms[:n_act], self.obstacles):
            oxy = d.geom_xpos[gid][:2]
            if np.linalg.norm(cyl_xy - oxy) < self.cfg.cylinder.radius + ob[2] - 0.001:
                return True
            if np.linalg.norm(tip_xy - oxy) < self.cfg.probe.radius + ob[2] - 0.001:
                return True
        return False

    def _servo(self, target_box_xy, steps):
        """Drive the tip toward ``target`` for one control period (dt)."""
        n_phys = max(1, int(round(self.cfg.control.dt / steps / self.cfg.control.sim_timestep)))
        tip_world = self.to_world(target_box_xy)
        for _ in range(steps):
            self._ik_update(tip_world)
            for _ in range(n_phys):
                mujoco.mj_step(self.model, self.data)
                if not self._collided and self._check_collision():
                    self._collided = True

    def _settle(self, target_box_xy, max_seconds=4.0, tol=1e-3):
        """Run the controller until the commanded pose has converged and the arm has caught up."""
        n_phys = max(1, int(round(self.cfg.control.dt / self.cfg.control.ik_substeps / self.cfg.control.sim_timestep)))
        tip_world = self.to_world(target_box_xy)
        n_iter = int(max_seconds / (n_phys * self.cfg.control.sim_timestep))
        for i in range(n_iter):
            err = self._ik_update(tip_world)
            for _ in range(n_phys):
                mujoco.mj_step(self.model, self.data)
            if err < tol and np.linalg.norm(self.data.qpos[:7] - self.q_cmd) < 2e-3 and i > 50:
                break

    def step(self, action) -> Obs:
        """Move the tip by ``action`` (dx, dy) in box metres over one control period."""
        cfg = self.cfg
        a = np.asarray(action, float).copy()
        n = np.linalg.norm(a)
        if n > cfg.control.max_step:
            a *= cfg.control.max_step / n
        lo = cfg.probe.radius + 0.002
        self.target = np.clip(self.target + a, lo, self.size - lo)
        self._collided = False
        self._servo(self.target, cfg.control.ik_substeps)
        # Compliant target: if the probe could not follow (blocked), the target
        # yields to where the probe actually is instead of accumulating error.
        tip = self.to_box(self.data.site_xpos[self.tcp_site])
        if np.linalg.norm(tip - self.target) > cfg.control.yield_dist:
            self.target = np.clip(tip, lo, self.size - lo)
        self.t += 1
        return self.observe()

    # ------------------------------------------------------------------ observe
    def observe(self) -> Obs:
        d, m = self.data, self.model
        finger = self.to_box(d.site_xpos[self.tcp_site])
        cyl_xyz = d.xpos[self.cyl_body]
        cyl = self.to_box(cyl_xyz)
        qw, qx, qy, qz = d.qpos[self.cyl_qpos + 3:self.cyl_qpos + 7]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        contact = False
        cyl_geom = m.geom("cylinder_geom").id
        hand_bodies = (m.body("left_finger").id, m.body("right_finger").id, m.body("hand").id)
        probe_geoms = {m.geom("probe").id} if self.cfg.probe.enabled else set()
        for i in range(d.ncon):
            c = d.contact[i]
            g = {c.geom1, c.geom2}
            touches_hand = bool(g & probe_geoms) or any(m.geom_bodyid[x] in hand_bodies for x in g)
            if cyl_geom in g and touches_hand:
                contact = True
        # Collision is accumulated at every physics substep of the last control step.
        collision = bool(self._collided or self._check_collision())
        success = bool(np.linalg.norm(cyl - self.goal) <= self.cfg.task.goal_tol) and not collision
        return Obs(finger=finger, cylinder=cyl, goal=self.goal.copy(), obstacles=self.obstacles.copy(),
                   cylinder_yaw=yaw, contact=contact, collision=collision, success=success, t=self.t,
                   extra=dict(cyl_z=float(cyl_xyz[2]), tip_z=float(d.site_xpos[self.tcp_site][2])))

    # ------------------------------------------------------------------ render
    def render(self, camera="top", width=None, height=None):
        width = width or self.cfg.render.width
        height = height or self.cfg.render.height
        key = f"{camera}_{width}x{height}"
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def replay_pose(self, finger_xy, cyl_xy):
        """Kinematic replay of a recorded step: cap teleported to its tracked position,
        probe marker drawn at the fingertip, real probe hidden. No physics stepping."""
        m, d, cfg = self.model, self.data, self.cfg
        ft = floor_z(cfg)
        q = self.cyl_qpos
        d.qpos[q:q + 3] = [*self.to_world(cyl_xy), ft + cfg.cylinder.height / 2]
        d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        d.qvel[self.cyl_dof:self.cyl_dof + 6] = 0
        mk = m.geom("probe_marker").id
        m.geom_pos[mk] = [*self.to_world(finger_xy), ft + cfg.probe.tip_height + m.geom_size[mk, 1] + cfg.probe.radius]
        m.geom_quat[mk] = [1, 0, 0, 0]
        if cfg.probe.enabled:
            m.geom_group[m.geom("probe").id] = 1       # hide the arm's probe in group-0 renders
        mujoco.mj_forward(m, d)

    def end_replay(self):
        m = self.model
        m.geom_pos[m.geom("probe_marker").id, 2] = PARK_Z
        if self.cfg.probe.enabled:
            m.geom_group[m.geom("probe").id] = 0
        mujoco.mj_forward(m, self.data)

    def render_overhead(self, size=256):
        """Square overhead image of the box with the robot arm hidden (only the probe,
        cap, obstacles, goal and box are drawn). Used as the observation for
        image-based policies, where the arm would otherwise occlude the cap."""
        key = f"overhead_{size}x{size}"
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, size, size)
        r = self._renderers[key]
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = 1          # scene geoms (box, cap, obstacles, goal, probe) are group 0
        r.update_scene(self.data, camera="overhead", scene_option=opt)
        # The box frame has y down the phone image (calibration corner order) while the
        # camera has y up: flip so overhead renders match the orientation of the clips.
        return np.ascontiguousarray(r.render()[::-1])

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
