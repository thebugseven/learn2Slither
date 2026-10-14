import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the AI agent for the Snake game.")
    parser.add_argument("--sessions", type=int, help="Number of sessions to run.", default=1000)
    parser.add_argument("--load", type=str, help="Path to load the model from.", default=None)
    parser.add_argument("--save", type=str, help="Path to save the model to.", default=None)
    parser.add_argument("--dontlearn", action="store_true", help="If set, the agent will not learn during evaluation.")


    return parser.parse_args()