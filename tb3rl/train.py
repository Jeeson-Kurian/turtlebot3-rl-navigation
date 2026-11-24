"""Train DQN or DDPG in the 2D simulator and log per-episode results to CSV.

    python -m tb3rl.train --algo dqn --episodes 600 --seed 0
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np

from .agents import DDPG, DQN
from .sim import DT, NavEnv


def make_agent(algo, obs_dim, seed, **hp):
    return (DQN if algo == "dqn" else DDPG)(obs_dim, seed=seed, **hp)


def run(algo, episodes, seed, out_dir="results", hp=None):
    hp = hp or {}
    env = NavEnv(seed=seed)
    agent = make_agent(algo, env.obs_dim, seed, **hp)
    rows = []
    for ep in range(episodes):
        s, total, done = env.reset(), 0.0, False
        while not done:
            a_store, a_env = agent.act(s)
            s2, r, done, info = env.step(a_env)
            agent.remember(s, a_store, r, s2, float(info["collision"] or info["goal"]))
            agent.learn()
            s, total = s2, total + r
        rows.append({"episode": ep, "return": round(total, 2), "steps": env.steps,
                     "time_s": env.steps * DT, "collision": int(info["collision"]),
                     "goal": int(info["goal"])})
        if ep % 50 == 0:
            last = rows[-50:]
            print(f"{algo} ep {ep:4d}  return {np.mean([x['return'] for x in last]):7.1f}  "
                  f"goal {np.mean([x['goal'] for x in last]):.2f}  "
                  f"crash {np.mean([x['collision'] for x in last]):.2f}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{algo}_seed{seed}")
    with open(path + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(path + ".pkl", "wb") as f:
        pickle.dump(agent.actor if algo == "ddpg" else agent.q, f)
    with open(path + "_config.json", "w") as f:
        json.dump({"algo": algo, "episodes": episodes, "seed": seed, **hp}, f, indent=2)
    return agent, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "ddpg"], required=True)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.algo, a.episodes, a.seed, a.out)
