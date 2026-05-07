import os
import pandas as pd
import numpy as np

# ==========================================
# 1. THE INFERENCE LEDGER (Live Routing)
# ==========================================
# Updated to reflect our active Maker/Checker stack
MODEL_REPUTATION_LEDGER = {
    "gpt": 0.92,      # GPT-4o-mini historical accuracy
    "llama": 0.85,    # Llama-3.2-90b historical accuracy
    "qwen": 0.94      # Qwen2-VL-72b historical accuracy
}

def calculate_market_consensus_risk(model_a_id: str, model_a_grade: int, model_a_conf: int, 
                                    model_b_id: str, model_b_grade: int, model_b_conf: int) -> tuple:
    """
    Calculates the market-weighted consensus risk between the Maker and Checker.
    Imported by `src/pipeline.py` to determine final clinical triage routing.
    """
    # Retrieve reputations (default to 0.5 if unregistered model is used)
    rep_a = MODEL_REPUTATION_LEDGER.get(model_a_id, 0.5)
    rep_b = MODEL_REPUTATION_LEDGER.get(model_b_id, 0.5)
    
    # Calculate weighted confidence bids
    weighted_conf_a = (model_a_conf / 100.0) * rep_a
    weighted_conf_b = (model_b_conf / 100.0) * rep_b
    
    # Calculate Base Risk
    base_risk = 1.0 - ((weighted_conf_a + weighted_conf_b) / (rep_a + rep_b))
    
    # Apply Mathematical Variance Penalty if models disagree on the diagnosis
    variance_penalty = 0.0
    if model_a_grade != model_b_grade:
        variance_penalty = 0.2 + (0.15 * ((rep_a + rep_b) / 2.0))
        
    # Calculate Final Risk Score and strictly cap between 0.0 and 1.0
    risk_score = max(0.0, min(1.0, base_risk + variance_penalty))
    
    # Clinical Triage Routing logic
    if risk_score <= 0.25:
        decision = "🟢 Safe / Auto-Triage"
    elif risk_score <= 0.55:
        decision = "🟡 Review Required"
    else:
        decision = "🔴 Unsafe / Hard Deferral"
        
    return (round(risk_score, 3), decision)

# ==========================================
# 2. THE OFFLINE CALIBRATION ENGINE (Historical Analysis)
# ==========================================
# TODO: Verify these paths match your local Antigravity workspace
RESULTS_CSV = "results.csv"
GROUND_TRUTH_CSV = "data/ground_truth/ground_truth.csv"

def calculate_optimal_threshold(maker_id: str):
    """
    Acts as the historical ledger. Calculates the confidence variance between 
    correct and incorrect diagnoses to derive a dynamic safety threshold.
    """
    print(f"📊 Analyzing Offline Ledger for Maker: {maker_id.upper()}")
    
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(GROUND_TRUTH_CSV):
        print("[Error] CSV files missing. Ensure pipeline and ground truth exist.")
        return

    try:
        results_df = pd.read_csv(RESULTS_CSV)
        truth_df = pd.read_csv(GROUND_TRUTH_CSV)
    except Exception as e:
        print(f"[Error] Failed to read CSVs: {e}")
        return

    # Merge results with ground truth on the filename to verify accuracy
    merged = pd.merge(results_df, truth_df, left_on='filename', right_on='image_filename', how='inner')
    
    if merged.empty:
        print("[Error] Merge failed. No matching filenames between results and ground truth.")
        return

    # Filter out API errors/failures (-1)
    merged = merged[merged['maker_grade'] != -1]
    merged['is_correct'] = merged['maker_grade'] == merged['true_grade']
    
    # Isolate populations to find the hallucination band
    correct_subset = merged[merged['is_correct'] == True]
    incorrect_subset = merged[merged['is_correct'] == False]
    
    accuracy = (len(correct_subset) / len(merged)) * 100
    mean_conf_correct = correct_subset['maker_confidence'].mean() if not correct_subset.empty else 0
    mean_conf_incorrect = incorrect_subset['maker_confidence'].mean() if not incorrect_subset.empty else 0
    calibration_gap = mean_conf_correct - mean_conf_incorrect
    
    print("\n--- HISTORICAL VARIANCE REPORT ---")
    print(f"Total Evaluated: {len(merged)}")
    print(f"Baseline Accuracy: {accuracy:.2f}%")
    print(f"Mean Confidence (Correct): {mean_conf_correct:.2f}%")
    print(f"Mean Confidence (Incorrect/Hallucination): {mean_conf_incorrect:.2f}%")
    print(f"Calibration Gap: {calibration_gap:.2f} points")
    
    # We set the new safety threshold just above the model's average hallucination confidence
    if mean_conf_incorrect > 0:
        suggested_threshold = min(int(np.ceil(mean_conf_incorrect)) + 2, 98)
    else:
        suggested_threshold = 95 
        
    print("\n--- ACTION REQUIRED ---")
    print(f"✅ Suggested Optimal Threshold for {maker_id.upper()}: {suggested_threshold}%")
    print(f"Update your `CALIBRATED_THRESHOLDS` dictionary in `src/pipeline.py` with:")
    print(f"'{maker_id}': {suggested_threshold}")

if __name__ == "__main__":
    # When executing this file directly, it runs the offline calibration.
    # Swap 'gpt' to 'llama' depending on which batch you just finished.
    calculate_optimal_threshold("gpt")