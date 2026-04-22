import pandas as pd
import os
import json
from pathlib import Path

GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth.csv")
PREDICTIONS_PATH = Path("output.csv")
CONFIG_PATH = Path("eval_config.json")

def main():
    if not GROUND_TRUTH_PATH.exists():
        print(f"Error: Ground truth not found at {GROUND_TRUTH_PATH}")
        return
        
    if not PREDICTIONS_PATH.exists():
        print(f"Error: Predictions not found at {PREDICTIONS_PATH}")
        return
        
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    gt_df = pd.read_csv(GROUND_TRUTH_PATH)
    
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    
    # Merge datasets on image_filename
    merged_df = pd.merge(pred_df, gt_df, on="image_filename", how="inner")
    
    if merged_df.empty:
        print("Error: No matching records found between predictions and ground truth.")
        return
        
    total_samples = len(merged_df)
    
    HITL_THRESHOLD = 85
    
    # Load config for model name
    model_name = "unknown"
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                model_name = config.get("model", config.get("openai_model", "unknown"))
        except Exception:
            pass

    # Unit Economics Engine
    pricing_per_1k = {
        "gpt-4o": 3.80,
        "gpt-4o-mini": 0.15,
        "gemini-1.5-pro": 3.50,
        "gemini-1.5-flash": 0.10
    }
    cost_per_1k = pricing_per_1k.get(model_name.lower(), 0.0)

    # Calculate accuracy
    # Make sure predictions and true_grade are cast to strings for fair comparison
    merged_df["prediction_str"] = merged_df["prediction"].astype(str).str.strip()
    merged_df["true_grade_str"] = merged_df["true_grade"].astype(str).str.strip()
    
    correct_matches = (merged_df["prediction_str"] == merged_df["true_grade_str"]).sum()
    accuracy = (correct_matches / total_samples) * 100

    # Calculate False Positive Rate
    merged_df["prediction_num"] = pd.to_numeric(merged_df["prediction"], errors='coerce')
    merged_df["true_grade_num"] = pd.to_numeric(merged_df["true_grade"], errors='coerce')
    
    true_0_mask = merged_df["true_grade_num"] == 0
    total_true_0s = true_0_mask.sum()
    
    fp_mask = true_0_mask & (merged_df["prediction_num"] >= 1)
    total_fps = fp_mask.sum()
    
    fpr = (total_fps / total_true_0s * 100) if total_true_0s > 0 else 0.0
    
    # Calculate HITL Triage Rate
    if "confidence_score" in merged_df.columns:
        merged_df["conf_num"] = pd.to_numeric(merged_df["confidence_score"], errors='coerce')
        human_review_required = (merged_df["conf_num"] < HITL_THRESHOLD).sum()
    else:
        human_review_required = 0
    
    hitl_triage_rate = (human_review_required / total_samples * 100) if total_samples > 0 else 0.0
    
    print("-" * 41)
    print("MEDVISION-BENCH: RUN REPORT")
    print("-" * 41)
    print(f"Model Evaluated: {model_name}")
    print(f"Total Images:    {total_samples}")
    print()
    print(f"1. Clinical Accuracy: {accuracy:.2f}%")
    print(f"2. False Positive Rate (Overthinking): {fpr:.2f}%")
    print(f"3. Est. Batch Cost per 1k Images: ${cost_per_1k:.2f}")
    print(f"5. HITL Triage Rate (<85% Conf): {hitl_triage_rate:.2f}%")
    print("-" * 41)

if __name__ == "__main__":
    main()
