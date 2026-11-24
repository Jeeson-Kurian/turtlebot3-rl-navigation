import unittest

import numpy as np

from tb3rl.agents import DDPG, DQN
from tb3rl.nn import MLP
from tb3rl.sim import MAX_RANGE, N_BEAMS, NavEnv


class TestMLP(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        for act in (None, "tanh"):
            net = MLP([4, 8, 3], out_act=act, seed=1)
            x = np.random.default_rng(0).normal(size=(5, 4))
            g_out = np.random.default_rng(1).normal(size=(5, 3))
            net.forward(x)
            grads, g_in = net.backward(g_out)
            loss = lambda: (net.forward(x) * g_out).sum()
            for p, g in zip(net.params(), grads):
                i = np.unravel_index(0, p.shape); eps = 1e-6
                p[i] += eps; up = loss(); p[i] -= 2 * eps; dn = loss(); p[i] += eps
                self.assertAlmostEqual((up - dn) / (2 * eps), g[i], places=4)
            x[0, 0] += 1e-6; up = loss(); x[0, 0] -= 2e-6; dn = loss(); x[0, 0] += 1e-6
            self.assertAlmostEqual((up - dn) / 2e-6, g_in[0, 0], places=4)


class TestSim(unittest.TestCase):
    def test_observation_shape_and_range(self):
        env = NavEnv(seed=3); s = env.reset()
        self.assertEqual(s.shape, (N_BEAMS + 2,))
        self.assertTrue(np.all(env.scan >= 0) and np.all(env.scan <= MAX_RANGE))

    def test_driving_into_wall_collides(self):
        env = NavEnv(seed=0, n_obstacles=0); env.reset()
        env.pos, env.yaw, env.goal = np.array([1.5, 0.0]), 0.0, np.array([-1.5, 1.5])
        env.prev_dist = np.hypot(*(env.goal - env.pos))
        for _ in range(50):
            _, r, done, info = env.step(0.0, 0.22)
            if done:
                break
        self.assertTrue(info["collision"]); self.assertEqual(r, -100.0)

    def test_reaching_goal(self):
        env = NavEnv(seed=0, n_obstacles=0); env.reset()
        env.pos, env.yaw, env.goal = np.array([0.0, 0.0]), 0.0, np.array([0.4, 0.0])
        env.prev_dist = 0.4
        for _ in range(10):
            _, r, done, info = env.step(0.0, 0.22)
            if done:
                break
        self.assertTrue(info["goal"]); self.assertEqual(r, 100.0)

    def test_progress_reward_sign(self):
        env = NavEnv(seed=0, n_obstacles=0); env.reset()
        env.pos, env.yaw, env.goal = np.array([0.0, 0.0]), 0.0, np.array([1.5, 0.0])
        env.prev_dist = 1.5
        _, r, _, _ = env.step(0.0, 0.2)
        self.assertGreater(r, 0)


class TestAgents(unittest.TestCase):
    def test_both_agents_learn_without_error(self):
        env = NavEnv(seed=0)
        for agent in (DQN(env.obs_dim, seed=0), DDPG(env.obs_dim, seed=0)):
            s = env.reset()
            for _ in range(1200):
                a, ae = agent.act(s)
                s2, r, d, info = env.step(ae)
                agent.remember(s, a, r, s2, float(info["collision"] or info["goal"]))
                loss = agent.learn()
                s = env.reset() if d else s2
            self.assertIsNotNone(loss); self.assertTrue(np.isfinite(loss))

    def test_ddpg_actions_bounded(self):
        agent = DDPG(NavEnv.obs_dim, seed=0)
        for _ in range(20):
            a, ae = agent.act(np.random.default_rng().normal(size=NavEnv.obs_dim))
            self.assertLessEqual(abs(a), 1.0)


if __name__ == "__main__":
    unittest.main()
