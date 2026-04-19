import pandas as pd
from pathlib import Path

GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth.csv")

def load_ground_truth():
    """Load the standardized ground truth labels."""
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found at {GROUND_TRUTH_PATH}")
    return pd.read_csv(GROUND_TRUTH_PATH)

def main():
    try:
        gt_df = load_ground_truth()
        print(f"Loaded ground truth with {len(gt_df)} records.")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
