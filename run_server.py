import argparse

from src.llm_runner.api import run_server
from src.llm_runner.model import LLMModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run_server(
        host=args.host,
        port=args.port,
        llm_model=LLMModel(args.model_name),
    )
