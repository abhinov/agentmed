import pandas as pd

# TODO: Ensure this matches your active results file
RESULTS_CSV = "results_gpt_qwen_images.csv"

print("🧹 Scanning ledger for failed API calls...")
df = pd.read_csv(RESULTS_CSV)

initial_count = len(df)

# Filter out any rows where the Maker completely failed to return a valid confidence score
clean_df = df[df['maker_confidence'] != -1]

failed_count = initial_count - len(clean_df)

if failed_count > 0:
    # Overwrite the CSV with only the successful rows
    clean_df.to_csv(RESULTS_CSV, index=False)
    print(f"✅ Removed {failed_count} corrupted/failed rows.")
    print("🚀 You can now re-run your pipeline to process these missing images.")
else:
    print("👍 Ledger is completely clean. No failed rows found.")
