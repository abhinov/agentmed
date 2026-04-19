import pandas as pd
import os

# --- CONFIGURATION ---
RAW_CSV_PATH = "idrid_labels.csv"
IMAGE_ID_COLUMN = "id_code"
TARGET_LABEL_COLUMN = "diagnosis"
APPEND_EXTENSION = ".jpg"
# ---------------------

OUTPUT_DIR = "data/ground_truth"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ground_truth.csv")

def main():
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Error: {RAW_CSV_PATH} not found.")
        return
        
    print(f"Reading {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    
    # Extract the necessary columns
    try:
        df = df[[IMAGE_ID_COLUMN, TARGET_LABEL_COLUMN]].copy()
    except KeyError as e:
        print(f"Error: Missing column in raw CSV. {e}")
        return
        
    # Append the extension to image IDs
    df[IMAGE_ID_COLUMN] = df[IMAGE_ID_COLUMN].astype(str) + APPEND_EXTENSION
    
    # Rename the columns strictly to expected format
    df.rename(columns={
        IMAGE_ID_COLUMN: "image_filename",
        TARGET_LABEL_COLUMN: "true_grade"
    }, inplace=True)
    
    # Ensure output directory exists (data/ground_truth)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the result
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully formatted labels and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
