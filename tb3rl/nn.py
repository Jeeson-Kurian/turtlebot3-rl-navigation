"""Small multilayer perceptron with manual backpropagation and Adam (NumPy only)."""
import numpy as np


class MLP:
    def __init__(self, sizes, out_act=None, seed=0, init_scale=None):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            scale = init_scale if (last and init_scale) else np.sqrt(2.0 / a)
            self.W.append(rng.normal(0, scale, (a, b)))
            self.b.append(np.zeros(b))
        self.out_act = out_act
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(z, 0)
            else:
                h = np.tanh(z) if self.out_act == "tanh" else z
            self.cache.append(h)
        return h

    def backward(self, grad_out):
        """Return (param grads, grad wrt input) for the last forward call."""
        n = len(self.W)
        gW, gb = [None] * n, [None] * n
        g = grad_out
        if self.out_act == "tanh":
            g = g * (1 - self.cache[-1] ** 2)
        for i in reversed(range(n)):
            h_in = self.cache[i]
            gW[i] = h_in.T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (self.cache[i] > 0)
        return gW + gb, g

    def step(self, grads, lr, clip=10.0):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        if norm > clip:
            grads = [g * clip / norm for g in grads]
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def copy_from(self, other, tau=1.0):
        for p, q in zip(self.params(), other.params()):
            p *= 1 - tau; p += tau * q
