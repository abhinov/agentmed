import os
import json
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# TODO: Ensure these paths match your local Antigravity workspace
RESULTS_CSV = "./results_llama_gpt_images.csv" 
GROUND_TRUTH_CSV = "./data/ground_truth/ground_truth.csv"
OUTPUT_REPORT = "./architecture_comparison_report.json"

def evaluate_architectures(results_path: str, truth_path: str):
    print("📊 Initializing Architecture Matrix Comparison...")
    
    if not os.path.exists(results_path) or not os.path.exists(truth_path):
        print(f"[Error] Missing required CSVs. Ensure both {results_path} and {truth_path} exist.")
        return

    # Load data
    try:
        results_df = pd.read_csv(results_path)
        truth_df = pd.read_csv(truth_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV data: {e}")
        return

    # Merge predictions with clinical ground truth on filename
    merged = pd.merge(results_df, truth_df, left_on='filename', right_on='image_filename', how='inner')
    
    if merged.empty:
        print("[Error] Merge failed. Filenames in results do not match ground truth.")
        return

    # Coerce to numeric for safe comparisons
    merged['maker_diagnosis'] = pd.to_numeric(merged['maker_diagnosis'], errors='coerce').fillna(-1).astype(int)
    if 'checker_diagnosis' in merged.columns:
        merged['checker_diagnosis'] = pd.to_numeric(merged['checker_diagnosis'], errors='coerce').fillna(-1).astype(int)

    # Filter out total API failures
    merged = merged[merged['maker_diagnosis'] != -1]
    total_evaluated = len(merged)

    # ==========================================
    # 1. SINGLE-AGENT BASELINE (GPT-4o-mini alone)
    # ==========================================
    merged['gpt_correct'] = merged['maker_diagnosis'] == merged['true_grade']
    baseline_accuracy = merged['gpt_correct'].mean()

    # ==========================================
    # 2. MULTI-AGENT CONSENSUS (GPT + Qwen)
    # ==========================================
    # Determine the final consensus grade dynamically
    def get_final_consensus(row):
        # If Qwen was triggered and successfully returned a grade, Qwen's second opinion overrides
        if row['checker_triggered'] == True and row['checker_diagnosis'] != -1:
            return row['checker_diagnosis']
        # Otherwise, the GPT Maker's original grade stands
        return row['maker_diagnosis']

    merged['final_consensus_grade'] = merged.apply(get_final_consensus, axis=1)
    merged['consensus_correct'] = merged['final_consensus_grade'] == merged['true_grade']
    consensus_accuracy = merged['consensus_correct'].mean()

    # ==========================================
    # 3. ESCALATION EFFICIENCY METRICS
    # ==========================================
    total_escalated = merged['checker_triggered'].sum()
    escalation_rate = total_escalated / total_evaluated

    # How often did Qwen successfully fix a hallucinated GPT diagnosis?
    fixed_by_qwen = merged[(merged['gpt_correct'] == False) & (merged['consensus_correct'] == True)]
    
    # How often did Qwen accidentally override a correct GPT diagnosis? (Over-correction)
    broken_by_qwen = merged[(merged['gpt_correct'] == True) & (merged['consensus_correct'] == False)]

    # Compile the final JSON Payload
    metrics_report = {
        "dataset_evaluated": "IDRiD",
        "total_images_processed": int(total_evaluated),
        "single_agent_baseline": {
            "model": "gpt-4o",
            "overall_accuracy": round(baseline_accuracy * 100, 2)
        },
        "multi_agent_consensus": {
            "architecture": "GPT (Maker) + Qwen (Checker)",
            "overall_accuracy": round(consensus_accuracy * 100, 2),
            "net_accuracy_gain": round((consensus_accuracy - baseline_accuracy) * 100, 2)
        },
        "routing_economics": {
            "total_checker_escalations": int(total_escalated),
            "escalation_rate_percentage": round(escalation_rate * 100, 2),
            "hallucinations_successfully_fixed": len(fixed_by_qwen),
            "unnecessary_over_corrections": len(broken_by_qwen)
        }
    }

    # Output strictly structured JSON report
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(metrics_report, f, indent=4)

    print("\n✅ Evaluation Complete. Results written to:", OUTPUT_REPORT)
    print(json.dumps(metrics_report, indent=4))

if __name__ == "__main__":
    evaluate_architectures(RESULTS_CSV, GROUND_TRUTH_CSV)