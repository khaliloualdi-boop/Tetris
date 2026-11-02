import time

import numpy as np

from tetris.agents import HeuristicAgent, RandomAgent
from tetris.env import TetrisEnv


def evaluate(agent, n_games: int = 20, max_pieces: int = 2000, render_mode: str | None = None) -> dict[str, list]:
    env = TetrisEnv(render_mode=render_mode, max_pieces=max_pieces)
    all_lines, all_pieces, all_truncated, all_rewards, all_times = [], [], [], [], []
    for game in range(n_games):
        obs, info = env.reset(seed=game)
        total_reward, start = 0.0, time.perf_counter()
        while True:
            a = agent.act(obs, info)
            obs, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if terminated or truncated:
                break
        all_lines.append(info["total_lines"])
        all_pieces.append(info["pieces_placed"])
        all_truncated.append(truncated)
        all_rewards.append(total_reward)
        all_times.append(time.perf_counter() - start)
    return {"lines": all_lines, "pieces": all_pieces, "truncated": all_truncated, "reward": all_rewards, "time": all_times}

def summarize(results: dict[str, list]) -> str:
    L = results["lines"]
    P = results["pieces"]
    T = results["truncated"]
    R = results["reward"]
    t = results["time"]

    return f"mean lines: {np.mean(L):.1f}, median lines: {np.median(L):.1f}, std: {np.std(L):.1f}, min / max: {np.min(L):.1f} / {np.max(L):.1f}, mean pieces: {np.mean(P):.1f}, truncated games: {sum([1 for value in T if value]):.1f}, mean reward: {np.mean(R):.1f}, mean time / game: {np.mean(t):.1f}"

def main() -> None:
    heuristic_agent = HeuristicAgent()
    random_agent = RandomAgent()

    print(summarize(evaluate(heuristic_agent)))
    print(summarize(evaluate(random_agent)))

if __name__ == "__main__":
    main()
