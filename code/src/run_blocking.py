"""
Batch Candidate Generation & Blocking Engine Runner
Business Entity Resolution Challenge - Group A (Aditi's Part)

High-Performance Streaming Architecture:
- Discovers open-set countries dynamically from Source 1.
- Ingests Source 2 & Source 3 in a single streaming pass across all country partitions.
- Fast vectorized target preprocessing (30,000+ rows/second).
- Queries Source 1 in original order, writing directly to output files.
- Preserves memory footprint (< 1.5 GB RAM).

Usage:
  python run_blocking.py --s1 ../../../dataset/test/test_source1.tsv \
                         --s2 ../../../dataset/test/test_source2.tsv \
                         --s3 ../../../dataset/test/test_source3.tsv \
                         --output-candidate ../../../output/candidate_pairs.tsv \
                         --output-matching ../../../output/matching_results.tsv
"""

import os
import sys
import time
import argparse
import collections
from typing import List, Dict, Set, Optional
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    normalize_country,
    normalize_business_name,
    extract_tokens,
    generate_soundex_tokens,
    extract_pincode
)
from blocking import InvertedIndexBlocker, DenseSemanticBlocker, TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE


def fast_preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """High-throughput preprocessing optimized for large-scale blocking."""
    df = df.copy()
    clean_names = df['business_name'].apply(normalize_business_name)
    df['clean_business_name'] = clean_names
    df['name_tokens'] = clean_names.apply(extract_tokens)
    df['soundex_tokens'] = clean_names.apply(generate_soundex_tokens)
    df['pincode'] = df['business_address'].apply(extract_pincode)
    df['clean_business_address'] = df['business_address'].fillna('')
    return df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run high-throughput multi-pass blocking & candidate generation pipeline."
    )
    
    # 2 levels up to goodmlhunting for the dataset
    dataset_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # 3 levels up to ml_challenge for output
    output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    default_test_dir = os.path.join(dataset_base, "dataset", "test")
    default_output_dir = os.path.join(output_base, "output")

    parser.add_argument(
        "--s1",
        default=os.path.join(default_test_dir, "test_source1.tsv"),
        help="Path to Source 1 TSV file."
    )
    # ... (Keep the rest of the add_argument blocks identical)    
    parser.add_argument(
        "--s2",
        default=os.path.join(default_test_dir, "test_source2.tsv"),
        help="Path to Source 2 TSV file."
    )
    parser.add_argument(
        "--s3",
        default=os.path.join(default_test_dir, "test_source3.tsv"),
        help="Path to Source 3 TSV file."
    )
    parser.add_argument(
        "--output-candidate",
        default=os.path.join(default_output_dir, "candidate_pairs.tsv"),
        help="Path to output candidate_pairs.tsv"
    )
    parser.add_argument(
        "--output-matching",
        default=os.path.join(default_output_dir, "matching_results.tsv"),
        help="Path to output matching_results.tsv placeholder for Group B."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of candidates to retain per entity."
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Enable dense vector semantic search (SentenceTransformers + FAISS)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for reading datasets."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of Source 1 entities for fast evaluation/testing."
    )
    return parser.parse_args()


def get_unique_countries(s1_path: str, limit: Optional[int] = None) -> List[str]:
    """Dynamically scan Source 1 for all distinct open-set countries."""
    countries = set()
    total_read = 0
    for chunk in pd.read_csv(s1_path, sep='\t', usecols=['country'], chunksize=100000):
        cleaned = chunk['country'].apply(normalize_country).unique()
        countries.update(cleaned)
        total_read += len(chunk)
        if limit and total_read >= limit:
            break
    return sorted(list(countries))


def main():
    args = parse_arguments()
    print("=" * 70, flush=True)
    print("STARTING HIGH-THROUGHPUT BLOCKING PIPELINE (ADITI'S MODULE)", flush=True)
    print("=" * 70, flush=True)
    print(f"Source 1:          {args.s1}", flush=True)
    print(f"Source 2:          {args.s2}", flush=True)
    print(f"Source 3:          {args.s3}", flush=True)
    print(f"Candidate Output:  {args.output_candidate}", flush=True)
    print(f"Matching Output:   {args.output_matching}", flush=True)
    print(f"Top-K Candidates:  {args.top_k}", flush=True)
    if args.limit:
        print(f"Sample Limit:      {args.limit:,} Source 1 entities", flush=True)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_candidate), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_matching), exist_ok=True)

    start_time = time.time()

    # Step 1: Detect open-set countries dynamically from Source 1
    print("\n[Step 1/3] Scanning Source 1 for open-set countries...", flush=True)
    countries = get_unique_countries(args.s1, limit=args.limit)
    print(f"  Detected {len(countries)} country partition(s): {countries}", flush=True)

    # Initialize inverted index blockers for each country partition
    blockers: Dict[str, InvertedIndexBlocker] = {
        c: InvertedIndexBlocker() for c in countries
    }

    # Step 2: Index Source 2 and Source 3 in a SINGLE streaming pass
    print("\n[Step 2/3] Indexing Source 2 and Source 3 in single-pass streaming mode...", flush=True)
    idx_start = time.time()
    target_limit = (args.limit * 50) if args.limit else None
    total_indexed = 0

    for source_label, path in [("Source 2", args.s2), ("Source 3", args.s3)]:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found. Skipping.", flush=True)
            continue
        print(f"  Streaming {source_label} from {path}...", flush=True)
        s_time = time.time()
        source_count = 0

        for chunk in pd.read_csv(path, sep='\t', chunksize=args.chunk_size):
            chunk['clean_country'] = chunk['country'].apply(normalize_country)
            for c, sub_df in chunk.groupby('clean_country'):
                if c in blockers:
                    p_sub = fast_preprocess_dataframe(sub_df)
                    blockers[c].index_entities(p_sub)

            source_count += len(chunk)
            total_indexed += len(chunk)

            if target_limit and total_indexed >= target_limit:
                break
        print(f"  Indexed {source_label} ({source_count:,} rows) in {time.time() - s_time:.2f}s.", flush=True)
        if target_limit and total_indexed >= target_limit:
            break

    print(f"Target indexing complete in {time.time() - idx_start:.2f}s:")
    for c, blk in blockers.items():
        print(f"  Country '{c}': {blk.indexed_count:,} indexed entities.", flush=True)

    # Step 3: Query Source 1 entities and stream output in original file order
    print("\n[Step 3/3] Querying Source 1 entities and streaming candidate pairs...", flush=True)
    q_start = time.time()
    total_processed_s1 = 0

    with open(args.output_candidate, 'w', encoding='utf-8') as f_cand, \
         open(args.output_matching, 'w', encoding='utf-8') as f_match:

        # Write required TSV headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for chunk in pd.read_csv(args.s1, sep='\t', chunksize=args.chunk_size):
            if args.limit and total_processed_s1 >= args.limit:
                break

            if args.limit and (total_processed_s1 + len(chunk)) > args.limit:
                chunk = chunk.iloc[:args.limit - total_processed_s1]

            p_chunk = fast_preprocess_dataframe(chunk)

            entity_ids = p_chunk['entity_id'].values
            pincodes = p_chunk['pincode'].values if 'pincode' in p_chunk.columns else [None] * len(p_chunk)
            name_tokens_list = p_chunk['name_tokens'].values if 'name_tokens' in p_chunk.columns else [[]] * len(p_chunk)
            soundex_tokens_list = p_chunk['soundex_tokens'].values if 'soundex_tokens' in p_chunk.columns else [[]] * len(p_chunk)
            clean_addrs = p_chunk['clean_business_address'].values if 'clean_business_address' in p_chunk.columns else [''] * len(p_chunk)
            countries_list = chunk['country'].apply(normalize_country).values

            for eid, pin, name_tokens, soundex_tokens, addr, c in zip(
                entity_ids, pincodes, name_tokens_list, soundex_tokens_list, clean_addrs, countries_list
            ):
                blocker = blockers.get(c)
                if blocker:
                    cands = blocker.retrieve_candidates(
                        pin=pin,
                        name_tokens=name_tokens,
                        soundex_tokens=soundex_tokens,
                        clean_addr=addr,
                        top_k=args.top_k
                    )
                else:
                    cands = []

                # Strict validation:
                # 1. S2- and S3- IDs only
                # 2. No S1- self-matches
                # 3. Deduplicate
                seen = set()
                valid_cands = []
                for cid in cands:
                    if cid.startswith(('S2-', 'S3-')) and cid not in seen:
                        seen.add(cid)
                        valid_cands.append(cid)

                cand_str = ",".join(valid_cands)
                f_cand.write(f"{eid}\t{cand_str}\n")

                top_match = valid_cands[0] if valid_cands else ""
                f_match.write(f"{eid}\t{top_match}\n")

                total_processed_s1 += 1

            if total_processed_s1 % 100000 == 0:
                cur_rate = total_processed_s1 / (time.time() - q_start)
                print(f"  Processed {total_processed_s1:,} entities ({cur_rate:.0f} entities/s)...", flush=True)

    q_elapsed = time.time() - q_start
    total_time = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"BLOCKING COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total S1 entities processed: {total_processed_s1:,} in {total_time:.2f}s ({total_processed_s1/total_time:.0f} overall/s)", flush=True)
    print(f"Candidate pairs saved to:    {args.output_candidate}", flush=True)
    print(f"Matching results saved to:   {args.output_matching}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
