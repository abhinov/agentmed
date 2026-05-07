"""
AgenticMed Evaluation Engine
Computes clinical metrics and safety calibration from pipeline outputs.
"""

import os
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_and_merge_data(inference_path: str, ground_truth_path: str) -> pd.DataFrame:
    """
    Robustly loads and securely merges the inference log with the ground truth.
    Performs type coercion to strictly align predictions with true labels.
    """
    if not os.path.exists(inference_path):
        raise FileNotFoundError(f"Inference log not found at: {inference_path}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth not found at: {ground_truth_path}")

    # Load Data
    inf_df = pd.read_csv(inference_path)
    gt_df = pd.read_csv(ground_truth_path)

    # Secure Inner Merge: guarantees we only score overlapping files
    merged_df = pd.merge(inf_df, gt_df, left_on='filename', right_on='image_filename', how='inner')

    # Detect diagnosis column name. Support both 'diagnosis_grade' and 'diagnosis'
    diag_col = 'diagnosis_grade' if 'diagnosis_grade' in merged_df.columns else 'diagnosis'
    
    if diag_col not in merged_df.columns or 'true_grade' not in merged_df.columns:
        raise ValueError("Missing essential diagnosis or true_grade columns for evaluation.")

    # Deterministic Type-Casting
    # Coerce to numeric, dropping outliers/errors, then casting exactly to integers.
    merged_df[diag_col] = pd.to_numeric(merged_df[diag_col], errors='coerce')
    merged_df['true_grade'] = pd.to_numeric(merged_df['true_grade'], errors='coerce')
    
    # Drop rows that couldn't be parsed into numerics
    initial_len = len(merged_df)
    merged_df.dropna(subset=[diag_col, 'true_grade'], inplace=True)
    dropped = initial_len - len(merged_df)
    if dropped > 0:
        print(f"Warning: Dropped {dropped} rows due to unparseable predictions.")

    # Cast to int for deterministic sklearn matching
    merged_df[diag_col] = merged_df[diag_col].astype(int)
    merged_df['true_grade'] = merged_df['true_grade'].astype(int)
    
    # Standardize column name internally
    merged_df.rename(columns={diag_col: 'diagnosis_grade'}, inplace=True)

    return merged_df

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Computes sklearn classification metrics."""
    y_true = df['true_grade']
    y_pred = df['diagnosis_grade']
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro averaging gives equal weight to each severity class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    return {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1_score": round(f1, 4)
    }

def calculate_calibration(df: pd.DataFrame) -> dict:
    """Calculates confidence calibration separating correct vs incorrect predictions."""
    if 'confidence_score' not in df.columns:
        return {"error": "confidence_score missing from inference log."}
        
    df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
    
    is_correct = (df['diagnosis_grade'] == df['true_grade'])
    
    correct_conf = df[is_correct]['confidence_score'].mean()
    incorrect_conf = df[~is_correct]['confidence_score'].mean()
    
    return {
        "mean_confidence_correct": round(float(correct_conf), 2) if pd.notna(correct_conf) else 0.0,
        "mean_confidence_incorrect": round(float(incorrect_conf), 2) if pd.notna(incorrect_conf) else 0.0,
        "calibration_gap": round(float(correct_conf - incorrect_conf), 2) if (pd.notna(correct_conf) and pd.notna(incorrect_conf)) else 0.0
    }

def generate_report(metrics: dict, calibration: dict, output_path: str):
    """Exports metrics to JSON and prints terminal summary."""
    report = {
        "clinical_metrics": metrics,
        "safety_calibration": calibration
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "="*50)
    print(" 🏥 AGENTICMED EVALUATION REPORT")
    print("="*50)
    print(f" Overall Accuracy:   {metrics.get('overall_accuracy', 0):.2%}")
    print(f" Macro Precision:    {metrics.get('macro_precision', 0):.4f}")
    print(f" Macro Recall:       {metrics.get('macro_recall', 0):.4f}")
    print(f" Macro F1-Score:     {metrics.get('macro_f1_score', 0):.4f}")
    print("-" * 50)
    print(f" Correct Pred. Conf:   {calibration.get('mean_confidence_correct', 0)}%")
    print(f" Incorrect Pred. Conf: {calibration.get('mean_confidence_incorrect', 0)}%")
    print(f" Calibration Gap:      {calibration.get('calibration_gap', 0)}")
    print("="*50)
    print(f"Report saved securely to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate clinical metrics.")
    parser.add_argument("--inference", type=str, default="results/inference_log.csv", help="Path to inference log")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth/ground_truth.csv", help="Path to ground truth")
    parser.add_argument("--output", type=str, default="results/metrics_report.json", help="Path for output JSON")
    
    args = parser.parse_args()
    
    try:
        # 1. Secure Join
        df = load_and_merge_data(args.inference, args.ground_truth)
        
        # 2. Extract Metrics
        metrics = calculate_metrics(df)
        
        # 3. Assess Safety Calibration
        calibration = calculate_calibration(df)
        
        # 4. Report
        generate_report(metrics, calibration, args.output)
        
    except Exception as e:
        print(f"Fatal Error during evaluation: {e}")

if __name__ == "__main__":
    main()
