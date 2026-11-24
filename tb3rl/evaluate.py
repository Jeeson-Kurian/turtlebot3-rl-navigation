"""Evaluate trained policies greedily on held-out arenas and plot learning curves.

    python -m tb3rl.evaluate --seeds 0 1 2 --episodes 200
"""
import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np

from .sim import DQN_ANGULAR, DT, MAX_ANG, NavEnv


def policy(algo, net):
    if algo == "dqn":
        return lambda s: DQN_ANGULAR[int(np.argmax(net.forward(s[None])[0]))]
    return lambda s: float(net.forward(s[None])[0, 0]) * MAX_ANG


def evaluate(algo, net, episodes, seed, noise=0.0):
    env, pi = NavEnv(seed=10_000 + seed, noise=noise), policy(algo, net)  # arenas never seen in training
    goals = crashes = 0
    times = []
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            s, _, done, info = env.step(pi(s))
        goals += info["goal"]; crashes += info["collision"]
        times.append(env.steps * DT)
    return {"success_rate": goals / episodes, "crash_rate": crashes / episodes,
            "mean_episode_time_s": float(np.mean(times))}


def survival(pi, runs, seed, cap_s=120.0):
    """Course metric 'time without crashing': keep driving to new goals until a crash or cap_s."""
    env = NavEnv(seed=20_000 + seed, max_steps=10**9)
    times, goals = [], []
    for _ in range(runs):
        s, n, t = env.reset(), 0, 0.0
        while t < cap_s:
            s, _, done, info = env.step(pi(s)); t += DT
            if info["collision"]:
                break
            if info["goal"]:
                n += 1
                while True:
                    env.goal = env._sample_free(0.4)
                    if np.hypot(*(env.goal - env.pos)) > 1.0:
                        break
                env.prev_dist = np.hypot(*(env.goal - env.pos)); s = env._obs()
        times.append(t); goals.append(n)
    return {"time_without_crash_s": float(np.mean(times)), "goals_per_run": float(np.mean(goals)),
            "runs_without_crash_in_120s": float(np.mean(np.array(times) >= cap_s))}


def random_baseline(episodes, seed):
    env, rng = NavEnv(seed=10_000 + seed), np.random.default_rng(seed)
    goals = crashes = 0; times = []
    for _ in range(episodes):
        env.reset(); done = False
        while not done:
            _, _, done, info = env.step(rng.uniform(-MAX_ANG, MAX_ANG))
        goals += info["goal"]; crashes += info["collision"]; times.append(env.steps * DT)
    rng2 = np.random.default_rng(seed + 1)
    out = {"success_rate": goals / episodes, "crash_rate": crashes / episodes,
           "mean_episode_time_s": float(np.mean(times))}
    out.update(survival(lambda s: rng2.uniform(-MAX_ANG, MAX_ANG), 50, seed))
    return out


def plot(results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for algo, col in (("dqn", "tab:blue"), ("ddpg", "tab:orange")):
        curves = []
        for f in sorted(glob.glob(f"{results_dir}/{algo}_seed*.csv")):
            rows = list(csv.DictReader(open(f)))
            curves.append(([float(r["return"]) for r in rows], [int(r["goal"]) for r in rows]))
        if not curves:
            continue
        w = 25
        for k, axis in enumerate(ax):
            sm = np.array([np.convolve(c[k], np.ones(w) / w, "valid") for c in curves])
            m, sd = sm.mean(0), sm.std(0)
            axis.plot(m, color=col, label=f"{algo.upper()} (mean of {len(curves)} seeds)")
            axis.fill_between(range(len(m)), m - sd, m + sd, color=col, alpha=0.2)
    ax[0].set(title="Episode return (25-episode moving average)", xlabel="Episode", ylabel="Return")
    ax[1].set(title="Goal reached (25-episode moving average)", xlabel="Episode", ylabel="Success rate")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/learning_curves.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--noise", type=float, nargs="*", default=[0.02, 0.05],
                   help="lidar noise std (m) for the robustness check")
    a = p.parse_args()
    summary = {}
    for algo in ("dqn", "ddpg"):
        per_seed = []
        for sd in a.seeds:
            path = f"{a.results}/{algo}_seed{sd}.pkl"
            if os.path.exists(path):
                net = pickle.load(open(path, "rb"))
                r = evaluate(algo, net, a.episodes, sd)
                r.update(survival(policy(algo, net), 50, sd))
                per_seed.append(r)
        if per_seed:
            summary[algo] = {k: [round(r[k], 3) for r in per_seed] for k in per_seed[0]}
    # robustness: same policies, noisy lidar they never saw in training
    summary["noise_robustness"] = {}
    for algo in ("dqn", "ddpg"):
        for n in a.noise:
            sr = [evaluate(algo, pickle.load(open(f"{a.results}/{algo}_seed{sd}.pkl", "rb")),
                           a.episodes, sd, noise=n)["success_rate"]
                  for sd in a.seeds if os.path.exists(f"{a.results}/{algo}_seed{sd}.pkl")]
            if sr:
                summary["noise_robustness"][f"{algo}_noise{n}"] = round(float(np.mean(sr)), 3)
    summary["random_policy"] = random_baseline(a.episodes, 0)
    json.dump(summary, open(f"{a.results}/evaluation.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    plot(a.results)
