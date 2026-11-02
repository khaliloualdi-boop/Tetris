import numpy as np

from tetris.env import TetrisEnv

N_GAMES = 3

def main() -> None:
    env = TetrisEnv(render_mode="human")
    rng = np.random.default_rng(0)

    try:
        for game in range(N_GAMES):
            _, info = env.reset(seed=game)
            while True:
                legal = np.flatnonzero(info["action_mask"])
                action = int(rng.choice(legal))
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            print(f"game {game}: pieces={info['pieces_placed']}, lines={info['total_lines']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
