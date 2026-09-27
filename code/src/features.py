import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from typing import Dict, List, Any

def extract_pairwise_features(
    s1_row: Dict[str, Any], 
    cand_row: Dict[str, Any],
    cand_rank: int = 1,
    emb_sim: float = 0.0
) -> Dict[str, float]:
    """Compute lexical, phonetic, postal, and dense similarity features."""
    name1 = str(s1_row.get('clean_business_name', ''))
    name2 = str(cand_row.get('clean_business_name', ''))
    addr1 = str(s1_row.get('clean_business_address', ''))
    addr2 = str(cand_row.get('clean_business_address', ''))
    
    # 1. String metrics (RapidFuzz)
    name_lev = fuzz.ratio(name1, name2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
    name_jw = distance.JaroWinkler.similarity(name1, name2)
    
    addr_lev = fuzz.ratio(addr1, addr2) / 100.0
    addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
    
    # 2. Pincode matching
    pin1 = str(s1_row.get('pincode', '')) if s1_row.get('pincode') else ''
    pin2 = str(cand_row.get('pincode', '')) if cand_row.get('pincode') else ''
    if pin1 and pin2:
        pin_match = 1.0 if pin1 == pin2 else 0.0
        pin_prefix = 1.0 if pin1[:3] == pin2[:3] else 0.0
    else:
        pin_match = -1.0  # Missing indicator
        pin_prefix = -1.0
        
    # 3. Numeric tokens in address (house/suite/street numbers)
    nums1 = set([t for t in addr1.split() if t.isdigit()])
    nums2 = set([t for t in addr2.split() if t.isdigit()])
    num_jaccard = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 | nums2) else 0.5

    return {
        'name_lev': name_lev,
        'name_token_sort': name_token_sort,
        'name_token_set': name_token_set,
        'name_jw': name_jw,
        'addr_lev': addr_lev,
        'addr_token_set': addr_token_set,
        'pin_match': pin_match,
        'pin_prefix': pin_prefix,
        'num_jaccard': num_jaccard,
        'emb_sim': emb_sim,
        'cand_rank': cand_rank
    }