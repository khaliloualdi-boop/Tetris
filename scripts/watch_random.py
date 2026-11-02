from tetris.agents import HeuristicAgent
from scripts.evaluate import evaluate
import numpy as np

from tetris.env import TetrisEnv

N_GAMES = 3

def main() -> None:
    heuristic_agent = HeuristicAgent()
    evaluate(heuristic_agent, render_mode="human")


if __name__ == "__main__":
    main()
