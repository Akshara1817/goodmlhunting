"""
Prediction Pipeline (predict.py)
Applies the trained model to candidate pairs and produces matching_results.tsv.
"""

import os
import argparse
import pandas as pd
import lightgbm as lgb
from features import extract_pairwise_features
from preprocessing import normalize_country, normalize_business_name, extract_tokens, generate_soundex_tokens, extract_pincode


def fast_preprocess(df):
    df = df.copy()
    df['clean_country'] = df['country'].apply(normalize_country)
    df['clean_business_name'] = df['business_name'].apply(normalize_business_name)
    df['clean_business_address'] = df['business_address'].fillna('')
    df['pincode'] = df['business_address'].apply(extract_pincode)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', default='../../student_resource/dataset/test/test_source1.tsv')
    parser.add_argument('--s2', default='../../student_resource/dataset/test/test_source2.tsv')
    parser.add_argument('--s3', default='../../student_resource/dataset/test/test_source3.tsv')
    parser.add_argument('--candidates', default='../../output/candidate_pairs.tsv')
    parser.add_argument('--output', default='../../output/matching_results.tsv')
    parser.add_argument('--model', default='lgbm_model.txt')
    parser.add_argument('--threshold-file', default='optimal_threshold.txt')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("STARTING INFERENCE PIPELINE")
    print("=" * 60)

    # 1. Load threshold
    threshold = 0.72
    if os.path.exists(args.threshold_file):
        with open(args.threshold_file, 'r') as f:
            threshold = float(f.read().strip())
    print(f"Using Decision Threshold: {threshold:.2f}")

    # 2. Load model
    print(f"Loading LightGBM model from {args.model}...")
    model = lgb.Booster(model_file=args.model)

    # 3. Load datasets
    print("Loading test datasets...")
    s1_raw = pd.read_csv(args.s1, sep='\t')
    if args.limit:
        s1_raw = s1_raw.iloc[:args.limit]
    s2_raw = pd.read_csv(args.s2, sep='\t')
    s3_raw = pd.read_csv(args.s3, sep='\t')

    s1_df = fast_preprocess(s1_raw).set_index('entity_id')
    s2_df = fast_preprocess(s2_raw).set_index('entity_id')
    s3_df = fast_preprocess(s3_raw).set_index('entity_id')
    targets_df = pd.concat([s2_df, s3_df])

    # 4. Stream through candidate pairs
    print(f"Reading candidates from {args.candidates}...")
    cand_df = pd.read_csv(args.candidates, sep='\t')
    if args.limit:
        cand_df = cand_df.iloc[:args.limit]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    match_results = []
    print("Scoring candidate pairs...")

    for _, row in cand_df.iterrows():
        s1_id = row['source1_entity_id']
        raw_cands = row['candidate_entity_ids']

        if pd.isna(raw_cands) or not str(raw_cands).strip():
            match_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue

        cand_list = [c.strip() for c in str(raw_cands).split(',') if c.strip()]
        if s1_id not in s1_df.index:
            match_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue

        s1_row = s1_df.loc[s1_id].to_dict()
        feature_rows = []
        valid_cands = []

        for rank, cid in enumerate(cand_list):
            if cid in targets_df.index:
                cand_row = targets_df.loc[cid].to_dict()
                feats = extract_pairwise_features(s1_row, cand_row, cand_rank=rank + 1)
                feature_rows.append(feats)
                valid_cands.append(cid)

        if not feature_rows:
            match_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue

        preds = model.predict(pd.DataFrame(feature_rows))
        matched = [valid_cands[i] for i, p in enumerate(preds) if p >= threshold]

        match_results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(matched)
        })

    out_df = pd.DataFrame(match_results)
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"Wrote {len(out_df):,} rows to {args.output}")


if __name__ == '__main__':
    main()