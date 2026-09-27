"""
Unified Training Script (train.py)
Supports both local testing (with --limit) and SageMaker container execution.
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from features import extract_pairwise_features
from blocking import InvertedIndexBlocker
from preprocessing import normalize_country, normalize_business_name, extract_tokens, generate_soundex_tokens, extract_pincode


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths configured to work both locally and in SageMaker containers
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../../../student_resource/dataset/train'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--limit', type=int, default=None, help='Limit S1 entities for local fast testing')
    parser.add_argument('--neg-ratio', type=int, default=4, help='Hard negatives per positive')
    return parser.parse_args()


def compute_macro_f05(y_true_dict, y_pred_dict):
    """Computes Macro F_0.5 score including singletons."""
    f05_scores = []
    for s1_id, true_set in y_true_dict.items():
        pred_set = y_pred_dict.get(s1_id, set())
        if len(true_set) == 0 and len(pred_set) == 0:
            f05_scores.append(1.0)
            continue
        if len(true_set) == 0 or len(pred_set) == 0:
            f05_scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = (0.25 * prec) + rec
        score = (1.25 * prec * rec) / denom if denom > 0 else 0.0
        f05_scores.append(score)
    return float(np.mean(f05_scores))


def fast_preprocess(df):
    df = df.copy()
    df['clean_country'] = df['country'].apply(normalize_country)
    df['clean_business_name'] = df['business_name'].apply(normalize_business_name)
    df['clean_business_address'] = df['business_address'].fillna('')
    df['pincode'] = df['business_address'].apply(extract_pincode)
    df['name_tokens'] = df['clean_business_name'].apply(extract_tokens)
    df['soundex_tokens'] = df['clean_business_name'].apply(generate_soundex_tokens)
    return df


def main():
    args = parse_args()
    print(f"Starting Training Pipeline | Train Dir: {args.train_dir} | Limit: {args.limit}")

    # 1. Load datasets
    s1_raw = pd.read_csv(os.path.join(args.train_dir, 'train_source1.tsv'), sep='\t')
    if args.limit:
        s1_raw = s1_raw.iloc[:args.limit]
    s2_raw = pd.read_csv(os.path.join(args.train_dir, 'train_source2.tsv'), sep='\t')
    s3_raw = pd.read_csv(os.path.join(args.train_dir, 'train_source3.tsv'), sep='\t')
    gt_df = pd.read_csv(os.path.join(args.train_dir, 'train_ground_truth.tsv'), sep='\t')

    # Build ground truth dictionary
    ground_truth = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
        ground_truth[s1_id] = set([m.strip() for m in matches if m.strip()])

    # Preprocess
    print("Preprocessing sources...")
    s1_df = fast_preprocess(s1_raw).set_index('entity_id')
    s2_df = fast_preprocess(s2_raw).set_index('entity_id')
    s3_df = fast_preprocess(s3_raw).set_index('entity_id')
    targets_df = pd.concat([s2_df, s3_df])

    # 2. Candidate generation for training set (Hard Negative Mining)
    print("Generating training candidates via blocking...")
    blockers = {}
    for c, sub_df in targets_df.groupby('clean_country'):
        blk = InvertedIndexBlocker()
        blk.index_entities(sub_df.reset_index())
        blockers[c] = blk

    pairs = []
    labels = []
    groups = []

    for s1_id, s1_row in s1_df.iterrows():
        c_code = s1_row['clean_country']
        blk = blockers.get(c_code)
        cands = blk.retrieve_candidates(
            pin=s1_row['pincode'],
            name_tokens=s1_row['name_tokens'],
            soundex_tokens=s1_row['soundex_tokens'],
            clean_addr=s1_row['clean_business_address'],
            top_k=20
        ) if blk else []

        true_matches = ground_truth.get(s1_id, set())

        # Include all ground truth positives
        for cid in true_matches:
            if cid in targets_df.index:
                pairs.append((s1_id, cid, 1))
                labels.append(1)
                groups.append(s1_id)

        # Include top hard negatives
        neg_count = 0
        for rank, cid in enumerate(cands):
            if cid not in true_matches and cid in targets_df.index:
                pairs.append((s1_id, cid, rank + 1))
                labels.append(0)
                groups.append(s1_id)
                neg_count += 1
                if neg_count >= args.neg_ratio:
                    break

    print(f"Total training pairs: {len(pairs):,} (Positives: {sum(labels):,}, Negatives: {len(labels) - sum(labels):,})")

    # 3. Extract pairwise features
    print("Extracting feature vectors...")
    feature_list = []
    for s1_id, cid, rank in pairs:
        row1 = s1_df.loc[s1_id].to_dict()
        row2 = targets_df.loc[cid].to_dict()
        feats = extract_pairwise_features(row1, row2, cand_rank=rank)
        feature_list.append(feats)

    X = pd.DataFrame(feature_list)
    y = np.array(labels)
    groups = np.array(groups)

    # 4. Stratified GroupKFold & Threshold Sweep
    print("Training LightGBM with 5-Fold GroupKFold...")
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof_preds = np.zeros(len(X))
    models = []

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    for fold, (t_idx, v_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_data = lgb.Dataset(X.iloc[t_idx], label=y[t_idx])
        val_data = lgb.Dataset(X.iloc[v_idx], label=y[v_idx], reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=800,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        oof_preds[v_idx] = model.predict(X.iloc[v_idx])
        models.append(model)

    # 5. Calibrate threshold for Macro F_0.5
    print("Optimizing threshold for Macro F_0.5...")
    best_thresh, best_f05 = 0.5, 0.0
    s1_keys = list(s1_df.index)

    for th in np.arange(0.50, 0.92, 0.02):
        pred_dict = {s1_id: set() for s1_id in s1_keys}
        for i, (s1_id, cid, _) in enumerate(pairs):
            if oof_preds[i] >= th:
                pred_dict[s1_id].add(cid)

        eval_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in s1_keys}
        score = compute_macro_f05(eval_gt, pred_dict)
        if score > best_f05:
            best_f05 = score
            best_thresh = th

    print(f"Optimal Threshold: {best_thresh:.2f} | Out-Of-Fold Macro F_0.5: {best_f05:.4f}")

    # 6. Save final model and threshold
    os.makedirs(args.model_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_dir, 'lgbm_model.txt')
    models[0].save_model(model_save_path)

    thresh_path = os.path.join(args.model_dir, 'optimal_threshold.txt')
    with open(thresh_path, 'w') as f:
        f.write(str(round(best_thresh, 4)))

    print(f"Model saved to {model_save_path}")
    print(f"Threshold saved to {thresh_path}")


if __name__ == '__main__':
    main()