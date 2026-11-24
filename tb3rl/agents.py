"""Double DQN (5 discrete angular velocities) and DDPG (continuous angular velocity)."""
import numpy as np

from .nn import MLP
from .sim import DQN_ANGULAR, MAX_ANG


class Replay:
    def __init__(self, obs_dim, act_dim, size=50_000, seed=0):
        self.s = np.zeros((size, obs_dim)); self.s2 = np.zeros((size, obs_dim))
        self.a = np.zeros((size, act_dim)); self.r = np.zeros(size); self.d = np.zeros(size)
        self.size, self.n, self.i = size, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k):
        idx = self.rng.integers(0, self.n, k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def huber_grad(err, delta=1.0):
    return np.clip(err, -delta, delta)


class DQN:
    """Double DQN with a soft-updated target network and epsilon-greedy exploration."""

    def __init__(self, obs_dim, hidden=(64, 64), lr=5e-4, gamma=0.99, batch=64, tau=0.01,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000, seed=0):
        sizes = [obs_dim, *hidden, len(DQN_ANGULAR)]
        self.q, self.q_t = MLP(sizes, seed=seed), MLP(sizes, seed=seed)
        self.q_t.copy_from(self.q)
        self.buf = Replay(obs_dim, 1, seed=seed)
        self.lr, self.gamma, self.batch, self.tau = lr, gamma, batch, tau
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.steps = 0
        self.rng = np.random.default_rng(seed + 1)

    @property
    def eps(self):
        f = min(self.steps / self.eps_decay, 1.0)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, s, explore=True):
        if explore and self.rng.random() < self.eps:
            a = int(self.rng.integers(len(DQN_ANGULAR)))
        else:
            a = int(np.argmax(self.q.forward(s[None])[0]))
        return a, DQN_ANGULAR[a]

    def remember(self, s, a, r, s2, d):
        self.buf.add(s, [a], r, s2, d)
        self.steps += 1

    def learn(self):
        if self.buf.n < 1000:
            return None
        s, a, r, s2, d = self.buf.sample(self.batch)
        a = a[:, 0].astype(int)
        a2 = np.argmax(self.q.forward(s2), 1)                      # online net selects
        q2 = self.q_t.forward(s2)[np.arange(len(a2)), a2]          # target net evaluates
        y = r + self.gamma * (1 - d) * q2
        q = self.q.forward(s)
        err = q[np.arange(len(a)), a] - y
        g = np.zeros_like(q)
        g[np.arange(len(a)), a] = huber_grad(err) / len(a)
        grads, _ = self.q.backward(g)
        self.q.step(grads, self.lr)
        self.q_t.copy_from(self.q, self.tau)
        return float(np.mean(err ** 2))


class DDPG:
    """DDPG with target networks, Gaussian exploration noise and soft updates."""

    def __init__(self, obs_dim, hidden=(64, 64), lr_actor=1e-4, lr_critic=5e-4, gamma=0.99,
                 batch=64, tau=0.005, noise_start=0.5, noise_end=0.1, noise_decay_steps=20_000,
                 seed=0):
        self.actor = MLP([obs_dim, *hidden, 1], out_act="tanh", seed=seed, init_scale=3e-3)
        self.actor_t = MLP([obs_dim, *hidden, 1], out_act="tanh", seed=seed, init_scale=3e-3)
        self.critic = MLP([obs_dim + 1, *hidden, 1], seed=seed + 7, init_scale=3e-3)
        self.critic_t = MLP([obs_dim + 1, *hidden, 1], seed=seed + 7, init_scale=3e-3)
        self.actor_t.copy_from(self.actor); self.critic_t.copy_from(self.critic)
        self.buf = Replay(obs_dim, 1, seed=seed)
        self.lr_a, self.lr_c, self.gamma, self.batch, self.tau = lr_actor, lr_critic, gamma, batch, tau
        self.n0, self.n1, self.nd = noise_start, noise_end, noise_decay_steps
        self.steps = 0
        self.rng = np.random.default_rng(seed + 1)

    @property
    def noise(self):
        f = min(self.steps / self.nd, 1.0)
        return self.n0 + f * (self.n1 - self.n0)

    def act(self, s, explore=True):
        a = float(self.actor.forward(s[None])[0, 0])
        if explore:
            a = float(np.clip(a + self.rng.normal(0, self.noise), -1, 1))
        return a, a * MAX_ANG

    def remember(self, s, a, r, s2, d):
        self.buf.add(s, [a], r, s2, d)
        self.steps += 1

    def learn(self):
        if self.buf.n < 1000:
            return None
        s, a, r, s2, d = self.buf.sample(self.batch)
        a2 = self.actor_t.forward(s2)
        y = r + self.gamma * (1 - d) * self.critic_t.forward(np.hstack([s2, a2]))[:, 0]
        q = self.critic.forward(np.hstack([s, a]))[:, 0]
        err = q - y
        grads, _ = self.critic.backward(huber_grad(err)[:, None] / len(err))
        self.critic.step(grads, self.lr_c)
        # actor: ascend Q(s, mu(s)) using dQ/da from the critic
        mu = self.actor.forward(s)
        self.critic.forward(np.hstack([s, mu]))
        _, g_in = self.critic.backward(-np.ones((len(s), 1)) / len(s))
        grads, _ = self.actor.backward(g_in[:, -1:])
        self.actor.step(grads, self.lr_a)
        self.actor_t.copy_from(self.actor, self.tau)
        self.critic_t.copy_from(self.critic, self.tau)
        return float(np.mean(err ** 2))
