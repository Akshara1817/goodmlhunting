import rapidfuzz

def get_lexical_features(s1_ascii: str, cand_ascii: str) -> dict:
    """Computes string similarity metrics between two romanized strings."""
    # Handle empty strings to prevent division by zero or errors
    if not s1_ascii or not cand_ascii:
        return {"fuzz_ratio": 0.0, "token_sort": 0.0, "token_set": 0.0}

    return {
        "fuzz_ratio": rapidfuzz.fuzz.ratio(s1_ascii, cand_ascii),
        "token_sort": rapidfuzz.fuzz.token_sort_ratio(s1_ascii, cand_ascii),
        "token_set": rapidfuzz.fuzz.token_set_ratio(s1_ascii, cand_ascii)
    }

def number_overlap_features(nums1: set, nums2: set) -> tuple:
    if not nums1 or not nums2:
        return 0.0, 0
    intersection = nums1.intersection(nums2)
    jaccard = len(intersection) / len(nums1.union(nums2))
    return jaccard, len(intersection)

# --- LOCAL TESTING BLOCK ---
if __name__ == "__main__":
    print("--- Testing Lexical Features ---")
    # Mock data representing processed ASCII names
    name1 = "ram marketting private limited"
    name2 = " राम marketing pvt ltd"
    
    lexical_scores = get_lexical_features(name1, name2)
    print(f"Comparing: '{name1}' vs '{name2}'")
    for metric, score in lexical_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\n--- Testing Number Overlap Features ---")
    # Mock data representing extracted address digits/pincodes
    address_nums1 = {"570", "13", "110045"}
    address_nums2 = {"570", "13", "110099"}
    
    jaccard, count = number_overlap_features(address_nums1, address_nums2)
    print(f"Comparing sets: {address_nums1} vs {address_nums2}")
    print(f"  Jaccard Similarity: {jaccard:.2f}")
    print(f"  Raw Overlap Count: {count}")