"""Draw a few greedy episodes of a trained policy in held-out arenas.

    python -m tb3rl.plot_trajectory --algo ddpg --seed 0
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import policy
from .sim import NavEnv


def main(algo, seed, results, n=4):
    net = pickle.load(open(f"{results}/{algo}_seed{seed}.pkl", "rb"))
    pi = policy(algo, net)
    env = NavEnv(seed=30_000 + seed)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax in axes:
        s, done, path = env.reset(), False, []
        while not done:
            path.append(env.pos.copy())
            s, _, done, info = env.step(pi(s))
        path.append(env.pos.copy()); path = np.array(path)
        for c, r in env.obstacles:
            ax.add_patch(plt.Circle(c, r, color="0.5"))
        ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=2)
        ax.plot(*path[0], "o", color="tab:green", label="start")
        ax.plot(*env.goal, "*", color="tab:red", ms=14, label="goal")
        ax.set_xlim(-env.half, env.half); ax.set_ylim(-env.half, env.half); ax.set_aspect("equal")
        ax.set_title("reached goal" if info["goal"] else ("crash" if info["collision"] else "timeout"))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{algo.upper()} (seed {seed}) on arenas it never saw during training")
    fig.tight_layout()
    fig.savefig(f"{results}/trajectories_{algo}.png", dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "ddpg"], default="ddpg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    a = p.parse_args()
    main(a.algo, a.seed, a.results)
