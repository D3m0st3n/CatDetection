"""One-off script to run the two-stage fine-tuning pipeline."""

import logging

from src.train import run_training
from src.utils import setup_logging

if __name__ == "__main__":
    setup_logging(logging.INFO)
    result = run_training()

    print()
    print("=== Training Complete ===")
    print(f"Stage 1 dir: {result.stage1_dir}")
    print(f"Stage 2 dir: {result.stage2_dir}")
    print(f"Best weights: {result.best_weights}")
    print(f"Results CSV: {result.results_csv}")
