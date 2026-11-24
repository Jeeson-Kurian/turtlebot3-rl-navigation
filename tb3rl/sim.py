"""Lightweight 2D TurtleBot3 simulator: differential drive, 2D lidar, circular obstacles.

Mirrors the observation and action interface of the ROS/Gazebo wrapper (ros_env.py), so the
same agents train here and in Gazebo. Physics is kinematic only: no slip, inertia or sensor
noise beyond optional Gaussian range noise.
"""
import numpy as np

N_BEAMS = 24          # downsampled from TurtleBot3's 360-beam LDS-01
MAX_RANGE = 3.5       # metres, LDS-01 maximum range
ROBOT_RADIUS = 0.105  # TurtleBot3 Burger footprint
MAX_LIN, MAX_ANG = 0.22, 2.0   # Burger velocity limits (m/s, rad/s)
DT = 0.25
COLLISION_DIST = ROBOT_RADIUS + 0.03
GOAL_DIST = 0.25
DQN_ANGULAR = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])


class NavEnv:
    """Square arena with random obstacles. obs = 24 scaled ranges + goal distance + goal angle."""

    obs_dim = N_BEAMS + 2

    def __init__(self, seed=0, half=2.0, n_obstacles=4, max_steps=300, noise=0.0):
        self.rng = np.random.default_rng(seed)
        self.half, self.n_obs, self.max_steps, self.noise = half, n_obstacles, max_steps, noise
        self.angles = np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)

    def _free(self, p, margin):
        if np.any(np.abs(p) > self.half - margin):
            return False
        return all(np.hypot(*(p - c)) > r + margin for c, r in self.obstacles)

    def _sample_free(self, margin):
        while True:
            p = self.rng.uniform(-self.half + margin, self.half - margin, 2)
            if self._free(p, margin):
                return p

    def reset(self):
        self.obstacles = []
        for _ in range(self.n_obs):
            c = self.rng.uniform(-self.half + 0.5, self.half - 0.5, 2)
            self.obstacles.append((c, self.rng.uniform(0.15, 0.3)))
        self.pos = self._sample_free(0.4)
        self.yaw = self.rng.uniform(-np.pi, np.pi)
        while True:
            self.goal = self._sample_free(0.4)
            if np.hypot(*(self.goal - self.pos)) > 1.0:
                break
        self.steps = 0
        self.prev_dist = np.hypot(*(self.goal - self.pos))
        return self._obs()

    def _scan(self):
        d = np.full(N_BEAMS, MAX_RANGE)
        dirs = np.stack([np.cos(self.angles + self.yaw), np.sin(self.angles + self.yaw)], 1)
        for k in range(2):  # arena walls
            for s in (-1, 1):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (s * self.half - self.pos[k]) / dirs[:, k]
                t[t <= 0] = np.inf
                d = np.minimum(d, t)
        for c, r in self.obstacles:  # ray-circle intersection
            oc = self.pos - c
            b = dirs @ oc
            disc = b * b - (oc @ oc - r * r)
            t = -b - np.sqrt(np.maximum(disc, 0))
            hit = (disc > 0) & (t > 0)
            d = np.where(hit, np.minimum(d, t), d)
        if self.noise:
            d = d + self.rng.normal(0, self.noise, N_BEAMS)
        return np.clip(d, 0, MAX_RANGE)

    def _obs(self):
        self.scan = self._scan()
        rel = self.goal - self.pos
        dist = np.hypot(*rel)
        ang = (np.arctan2(rel[1], rel[0]) - self.yaw + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([self.scan / MAX_RANGE, [min(dist / 4.0, 1.0), ang / np.pi]])

    def step(self, ang_vel, lin_vel=0.15):
        ang_vel = float(np.clip(ang_vel, -MAX_ANG, MAX_ANG))
        lin_vel = float(np.clip(lin_vel, 0.0, MAX_LIN))
        self.yaw += ang_vel * DT
        self.pos = self.pos + lin_vel * DT * np.array([np.cos(self.yaw), np.sin(self.yaw)])
        self.steps += 1
        obs = self._obs()
        dist = np.hypot(*(self.goal - self.pos))
        reward = 10.0 * (self.prev_dist - dist) - 0.01
        self.prev_dist = dist
        info = {"collision": False, "goal": False}
        done = False
        if self.scan.min() < COLLISION_DIST or not self._free(self.pos, ROBOT_RADIUS):
            reward, done, info["collision"] = -100.0, True, True
        elif dist < GOAL_DIST:
            reward, done, info["goal"] = 100.0, True, True
        elif self.steps >= self.max_steps:
            done = True
        return obs, reward, done, info
