"""
Blocking & Candidate Generation Pipeline Module (Group A - Aditi's Part)
Business Entity Resolution Challenge

This module implements high-recall, scalable multi-pass candidate generation:
1. Open-Set Country Partitioning:
   - Evaluates entities strictly within their country partition (US, India, France, etc.)
   - Dynamic country handling without hardcoding.
2. Pass 1: Inverted Index Multi-Pass Rule-Based Blocking:
   - Pincode / postal code indexing
   - Phonetic Soundex token hashing for typo resilience
   - High-precision name token overlap with inverted frequency weighting
   - Address locality & street token matching
3. Pass 2: Dense Vector Semantic Blocking:
   - Lightweight SentenceTransformer embeddings (e.g., all-MiniLM-L6-v2)
   - FAISS cosine similarity (IndexFlatIP with normalized vectors)
   - Approximate Nearest Neighbors (ANN) per country partition
4. Candidate Assembly & Output Generation:
   - Unified deduplicated union of Pass 1 and Pass 2 candidates
   - Strict validation: S2- and S3- IDs only, no S1 self-matches, no intra-list duplicates
   - Output formatted for output/candidate_pairs.tsv
"""

import re
import time
import collections
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def extract_address_tokens(addr_str: Any) -> List[str]:
    """Extract key address tokens (street names, house/plot numbers, locality)."""
    if not addr_str or pd.isna(addr_str):
        return []
    tokens = str(addr_str).lower().split()
    return [t for t in tokens if len(t) > 2 or t.isdigit()]


class InvertedIndexBlocker:
    """
    High-performance multi-pass inverted index blocker supporting pincode,
    name tokens, phonetic soundex tokens, and address tokens.
    """

    def __init__(self, max_posting_size: int = 2500):
        self.max_posting_size = max_posting_size
        self.pincode_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.token_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.soundex_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.addr_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.indexed_count = 0

    def index_entities(self, df: pd.DataFrame):
        """Fast vectorized indexing over target entities (Source 2 and 3)."""
        entity_ids = df['entity_id'].values
        pincodes = df['pincode'].values if 'pincode' in df.columns else [None] * len(df)
        name_tokens_list = df['name_tokens'].values if 'name_tokens' in df.columns else [[]] * len(df)
        soundex_tokens_list = df['soundex_tokens'].values if 'soundex_tokens' in df.columns else [[]] * len(df)
        clean_addrs = df['clean_business_address'].values if 'clean_business_address' in df.columns else [''] * len(df)

        for eid, pin, name_tokens, soundex_tokens, addr in zip(
            entity_ids, pincodes, name_tokens_list, soundex_tokens_list, clean_addrs
        ):
            if not (eid.startswith('S2-') or eid.startswith('S3-')):
                continue

            if pin and pd.notna(pin):
                if len(self.pincode_index[str(pin)]) < self.max_posting_size:
                    self.pincode_index[str(pin)].append(eid)

            if isinstance(name_tokens, list):
                for tok in name_tokens:
                    if len(self.token_index[tok]) < self.max_posting_size:
                        self.token_index[tok].append(eid)

            if isinstance(soundex_tokens, list):
                for snd in soundex_tokens:
                    if len(self.soundex_index[snd]) < self.max_posting_size:
                        self.soundex_index[snd].append(eid)

            addr_tokens = extract_address_tokens(addr)
            for atok in addr_tokens:
                if len(self.addr_index[atok]) < 250:
                    self.addr_index[atok].append(eid)

            self.indexed_count += 1

    def retrieve_candidates(
        self,
        pin: Any,
        name_tokens: List[str],
        soundex_tokens: List[str],
        clean_addr: str,
        top_k: int = 25
    ) -> List[str]:
        """
        Retrieves top candidate entity IDs for a single Source 1 query entity
        using weighted frequency scoring.
        """
        scores: Dict[str, float] = collections.defaultdict(float)

        # 1. Pincode match
        if pin and pd.notna(pin) and str(pin) in self.pincode_index:
            postings = self.pincode_index[str(pin)]
            if len(postings) <= self.max_posting_size:
                w = 1.5 / (1.0 + len(postings) ** 0.4)
                for eid in postings:
                    scores[eid] += w

        # 2. Name token overlap (higher weight, inversely proportional to frequency)
        if isinstance(name_tokens, list):
            for tok in name_tokens:
                if tok in self.token_index:
                    postings = self.token_index[tok]
                    if len(postings) <= self.max_posting_size:
                        w = 2.5 / (1.0 + len(postings) ** 0.5)
                        for eid in postings:
                            scores[eid] += w

        # 3. Soundex phonetic token overlap
        if isinstance(soundex_tokens, list):
            for snd in soundex_tokens:
                if snd in self.soundex_index:
                    postings = self.soundex_index[snd]
                    if len(postings) <= self.max_posting_size:
                        w = 1.0 / (1.0 + len(postings) ** 0.5)
                        for eid in postings:
                            scores[eid] += w

        # 4. Address token overlap
        addr_tokens = extract_address_tokens(clean_addr)
        for atok in addr_tokens:
            if atok in self.addr_index:
                postings = self.addr_index[atok]
                if len(postings) <= 200:  # Avoid overly common street words
                    w = 1.8 / (1.0 + len(postings) ** 0.5)
                    for eid in postings:
                        scores[eid] += w

        if not scores:
            return []

        # Sort candidate entities by aggregate score descending
        sorted_candidates = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_candidates[:top_k]


class DenseSemanticBlocker:
    """
    Pass 2: Dense Vector Semantic Blocking using SentenceTransformer & FAISS.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[Any] = None
        self.index: Optional[Any] = None
        self.target_ids: List[str] = []

    def _get_model(self):
        if self._model is None and TRANSFORMERS_AVAILABLE:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(self, df: pd.DataFrame):
        """Generates dense vector embeddings and indexes them using FAISS."""
        model = self._get_model()
        if model is None or not FAISS_AVAILABLE:
            return

        valid_df = df[df['entity_id'].str.startswith(('S2-', 'S3-'))].copy()
        if len(valid_df) == 0:
            return

        self.target_ids = valid_df['entity_id'].tolist()
        texts = (
            valid_df['clean_business_name'].fillna('') + ' ' + valid_df['clean_business_address'].fillna('')
        ).tolist()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product on normalized vectors = Cosine Similarity
        self.index.add(embeddings)

    def retrieve_candidates(self, query_df: pd.DataFrame, top_k: int = 15) -> Dict[str, List[str]]:
        """Retrieves Top-K nearest neighbors for Source 1 query entities."""
        results: Dict[str, List[str]] = collections.defaultdict(list)
        model = self._get_model()
        if model is None or self.index is None or len(self.target_ids) == 0:
            return results

        texts = (
            query_df['clean_business_name'].fillna('') + ' ' + query_df['clean_business_address'].fillna('')
        ).tolist()

        query_embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        actual_k = min(top_k, len(self.target_ids))
        distances, indices = self.index.search(query_embeddings, actual_k)

        for i, s1_id in enumerate(query_df['entity_id']):
            for idx in indices[i]:
                if 0 <= idx < len(self.target_ids):
                    results[s1_id].append(self.target_ids[idx])

        return results


class EntityResolutionBlockingEngine:
    """
    Group A Candidate Generation Pipeline Engine:
    Coordinates Open-Set Country Partitioning, Pass 1 Rule-Based Blocking,
    Pass 2 Dense Vector Blocking, and Candidate Deduplication & Formatting.
    """

    def __init__(
        self,
        top_k_rule: int = 25,
        top_k_dense: int = 10,
        enable_dense: bool = True,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        self.top_k_rule = top_k_rule
        self.top_k_dense = top_k_dense
        self.enable_dense = enable_dense and TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
        self.model_name = model_name

    def process_partition(
        self,
        country: str,
        s1_partition: pd.DataFrame,
        targets_partition: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Processes candidate generation within a single country partition."""
        partition_candidates: Dict[str, List[str]] = collections.defaultdict(list)
        if len(s1_partition) == 0:
            return partition_candidates

        # Pass 1: Inverted Index Rule-Based Blocking
        rule_blocker = InvertedIndexBlocker()
        rule_blocker.index_entities(targets_partition)

        s1_pins = s1_partition['pincode'].values if 'pincode' in s1_partition.columns else [None] * len(s1_partition)
        s1_names = s1_partition['name_tokens'].values if 'name_tokens' in s1_partition.columns else [[]] * len(s1_partition)
        s1_soundex = s1_partition['soundex_tokens'].values if 'soundex_tokens' in s1_partition.columns else [[]] * len(s1_partition)
        s1_addrs = s1_partition['clean_business_address'].values if 'clean_business_address' in s1_partition.columns else [''] * len(s1_partition)

        for s1_id, pin, name_tokens, soundex_tokens, addr in zip(
            s1_partition['entity_id'].values, s1_pins, s1_names, s1_soundex, s1_addrs
        ):
            cands = rule_blocker.retrieve_candidates(
                pin=pin,
                name_tokens=name_tokens,
                soundex_tokens=soundex_tokens,
                clean_addr=addr,
                top_k=self.top_k_rule
            )
            partition_candidates[s1_id].extend(cands)

        # Pass 2: Dense Vector Semantic Blocking (for partitions within memory capacity)
        if self.enable_dense and len(targets_partition) > 0 and len(targets_partition) <= 150000:
            dense_blocker = DenseSemanticBlocker(model_name=self.model_name)
            dense_blocker.build_index(targets_partition)
            dense_results = dense_blocker.retrieve_candidates(s1_partition, top_k=self.top_k_dense)
            for s1_id, d_cands in dense_results.items():
                partition_candidates[s1_id].extend(d_cands)

        # Deduplicate and validate candidate IDs
        final_partition: Dict[str, List[str]] = {}
        for s1_id in s1_partition['entity_id']:
            raw_cands = partition_candidates.get(s1_id, [])
            seen: Set[str] = set()
            clean_cands: List[str] = []
            for cid in raw_cands:
                if cid.startswith(('S2-', 'S3-')) and cid not in seen:
                    seen.add(cid)
                    clean_cands.append(cid)
            final_partition[s1_id] = clean_cands[:self.top_k_rule + self.top_k_dense]

        return final_partition

    def run_blocking(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Executes end-to-end multi-pass blocking across all country partitions.
        Returns DataFrame with columns ['source1_entity_id', 'candidate_entity_ids'].
        """
        print(f"Starting Multi-Pass Blocking on {len(s1_df):,} Source 1 entities...")
        targets_df = pd.concat([s2_df, s3_df], ignore_index=True)

        countries = s1_df['clean_country'].unique()
        print(f"Identified {len(countries)} country partition(s): {list(countries)}")

        all_candidate_pairs: List[Dict[str, str]] = []

        for country in countries:
            print(f"\nProcessing Country Partition: {country}...")
            s1_sub = s1_df[s1_df['clean_country'] == country]
            targets_sub = targets_df[targets_df['clean_country'] == country]
            print(f"  S1 Queries: {len(s1_sub):,}, S2/S3 Targets: {len(targets_sub):,}")

            part_start = time.time()
            part_cands = self.process_partition(country, s1_sub, targets_sub)
            print(f"  Partition completed in {time.time() - part_start:.2f} seconds.")

            for s1_id in s1_sub['entity_id']:
                c_list = part_cands.get(s1_id, [])
                all_candidate_pairs.append({
                    'source1_entity_id': s1_id,
                    'candidate_entity_ids': ",".join(c_list)
                })

        output_df = pd.DataFrame(all_candidate_pairs)
        print(f"\nBlocking completed: {len(output_df):,} rows generated.")
        return output_df


if __name__ == '__main__':
    print("Testing Blocking Engine on dummy data...")
    dummy_s1 = pd.DataFrame([
        {
            "entity_id": "S1-0001",
            "clean_country": "US",
            "clean_business_name": "holloway peak seafood",
            "clean_business_address": "105 elm street morganton nc 28655",
            "pincode": "28655",
            "name_tokens": ["holloway", "peak", "seafood"],
            "soundex_tokens": ["H400", "P200", "S130"]
        }
    ])
    dummy_s2 = pd.DataFrame([
        {
            "entity_id": "S2-0001",
            "clean_country": "US",
            "clean_business_name": "holloway peak foods",
            "clean_business_address": "105 elm st morganton nc 28655",
            "pincode": "28655",
            "name_tokens": ["holloway", "peak", "foods"],
            "soundex_tokens": ["H400", "P200", "F320"]
        }
    ])
    dummy_s3 = pd.DataFrame([
        {
            "entity_id": "S3-0001",
            "clean_country": "US",
            "clean_business_name": "holloway seafood inc",
            "clean_business_address": "elm street morganton",
            "pincode": None,
            "name_tokens": ["holloway", "seafood"],
            "soundex_tokens": ["H400", "S130"]
        }
    ])

    engine = EntityResolutionBlockingEngine(top_k_rule=10, top_k_dense=5, enable_dense=False)
    results = engine.run_blocking(dummy_s1, dummy_s2, dummy_s3)
    print("\nCandidate Pairs Output:")
    print(results.to_csv(sep='\t', index=False))
