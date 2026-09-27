"""
Batch Data Preprocessing Pipeline Runner
Business Entity Resolution Challenge - Group A (Aastha)
"""

import os
import time
import pandas as pd
from preprocessing import preprocess_dataframe


def process_file(input_path: str, output_path: str):
    """Processes a single dataset TSV file and saves the preprocessed output."""
    print(f"Loading {input_path}...")
    start_time = time.time()
    
    # Read TSV with explicit tab separator
    df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")
    
    # Preprocess
    p_start = time.time()
    processed_df = preprocess_dataframe(df)
    print(f"Preprocessed {len(df):,} rows in {time.time() - p_start:.2f} seconds.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save output as tab-separated file
    s_start = time.time()
    processed_df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to {output_path} in {time.time() - s_start:.2f} seconds.\n")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    train_dir = os.path.join(base_dir, "dataset", "train")
    output_dir = os.path.join(base_dir, "dataset", "preprocessed")
    
    files_to_process = [
        ("train_source1.tsv", os.path.join(output_dir, "clean_train_source1.tsv")),
        ("train_source2.tsv", os.path.join(output_dir, "clean_train_source2.tsv")),
        ("train_source3.tsv", os.path.join(output_dir, "clean_train_source3.tsv")),
    ]
    
    print("=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE (AASTHA'S PART)")
    print("=" * 60)
    
    for filename, out_path in files_to_process:
        in_path = os.path.join(train_dir, filename)
        if os.path.exists(in_path):
            process_file(in_path, out_path)
        else:
            print(f"Warning: File not found: {in_path}")

    print("Data preprocessing complete! Clean datasets ready for Aditi's Blocking Engine.")


if __name__ == '__main__':
    main()
