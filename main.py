"""Run the complete student performance project in order."""

from __future__ import annotations

import sys


def main() -> int:
    stages = [
        ("1/6 Download and validate dataset", "src.download_data", "download_data"),
        ("2/6 Explore data and create visualizations", "src.eda", "run_eda"),
        ("3/6 Preprocess and split data", "src.preprocessing", "load_and_prepare"),
        ("4/6 Train and select the neural network", "src.train_model", "train_model"),
        ("5/6 Evaluate on held-out test data", "src.evaluate_model", "evaluate_model"),
        ("6/6 Generate editable project report", "src.generate_report", "generate_report"),
    ]
    for label, module_name, function_name in stages:
        print(f"\n{label}")
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)()
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            print(f"Project stopped: {error}", file=sys.stderr)
            return 1
    print("\nProject completed. See outputs/ and report/ for generated results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
