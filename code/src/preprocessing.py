"""
Data Preprocessing & String Normalization Module (Group A - Aastha's Part)
Business Entity Resolution Challenge

This module handles:
1. Business name cleaning, legal suffix removal (US, India, France, etc.), DBA/trade name handling, and URL stripping.
2. Address parsing: street/unit abbreviation normalization, PIN/postal code extraction.
3. Open-set country normalization.
4. Phonetic Soundex token generation & normalized token set extraction for candidate blocking.
"""

import re
import unicodedata
import pandas as pd
from typing import Dict, List, Optional


# Comprehensive legal entity suffixes across global business entities (US, India, France, etc.)
LEGAL_SUFFIXES = [
    r'\bprivate limited\b', r'\bpvt limited\b', r'\bpvt ltd\b', r'\bpvt\b',
    r'\blimited\b', r'\bltd\b',
    r'\bcorporation\b', r'\bcorp\b',
    r'\bincorporated\b', r'\binc\b',
    r'\blimited liability company\b', r'\bllc\b',
    r'\blimited liability partnership\b', r'\bllp\b',
    r'\bpublic limited company\b', r'\bplc\b',
    r'\bcompany\b', r'\bco\b',
    r'\bgmbh\b', r'\bsa\b', r'\bsrl\b', r'\bsarl\b', r'\bsas\b', r'\beurl\b', r'\bsci\b',
    r'\bholding\b', r'\bholdings\b', r'\bgroup\b',
    r'\bservices\b', r'\benterprises\b', r'\bsolutions\b', r'\btechnologies\b',
    r'\bindustries\b', r'\btraders\b', r'\bproprietorship\b',
    r'\bpc\b', r'\blp\b'
]

LEGAL_SUFFIX_REGEX = re.compile(r'|'.join(LEGAL_SUFFIXES), re.IGNORECASE)

# Trade name / DBA patterns (Doing Business As)
DBA_REGEX = re.compile(r'\b(d/?b/?a|doing business as|t/a|trading as)\b.*$', re.IGNORECASE)

# Address abbreviation mappings
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bbldg\b': 'building',
    r'\bfl\b': 'floor',
    r'\bflr\b': 'floor',
    r'\bopp\b': 'opposite',
    r'\bb/h\b': 'behind',
    r'\bbh\b': 'behind',
    r'\bnr\b': 'near',
    r'\bh\s*no\b': 'houseno',
    r'\bflat\s*no\b': 'flatno',
    r'\bplot\s*no\b': 'plotno',
    r'\bkh\s*no\b': 'khasrano',
    r'\bno\b': 'number',
    r'\bp\s*o\s*box\b': 'pobox',
}

ADDR_REGEX_PATTERNS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBREVIATIONS.items()]
URL_REGEX = re.compile(r'\|?\s*(https?://\S+|www\.\S+|\b\S+\.(com|in|org|net|co|io)\b)', re.IGNORECASE)
PINCODE_REGEX = re.compile(r'\b(\d{6}|\d{5})\b')

# Soundex mapping dict
SOUNDEX_MAP = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}


def compute_soundex(word: str) -> str:
    """Computes standard 4-character Soundex code for a single word."""
    clean_word = re.sub(r'[^A-Za-z]', '', word).upper()
    if not clean_word:
        return ""
    
    first_char = clean_word[0]
    digits = [SOUNDEX_MAP.get(c, '0') for c in clean_word]
    
    # Collapse adjacent duplicates
    collapsed = [digits[0]]
    for d in digits[1:]:
        if d != collapsed[-1]:
            collapsed.append(d)
            
    res = [first_char] + [d for d in collapsed[1:] if d != '0']
    code = "".join(res)[:4]
    return code.ljust(4, '0')


def generate_soundex_tokens(text: str) -> List[str]:
    """Generates Soundex phonetic codes for all significant words in a text string."""
    words = re.findall(r'\b[A-Za-z]{2,}\b', text)
    soundex_codes = [compute_soundex(w) for w in words]
    return [c for c in soundex_codes if c]


def normalize_country(country_str: str) -> str:
    """Standardize country string dynamically (supports open-set countries like US, India, France)."""
    if pd.isna(country_str) or not str(country_str).strip():
        return "UNKNOWN"
    return str(country_str).strip().upper()


def normalize_business_name(name_str: str) -> str:
    """
    Build clean text pipelines for business names:
    - Strips URLs and domain mentions
    - Strips DBA/trade names
    - Strips legal suffixes (Pvt Ltd, Corp, Inc, LLC, SARL, etc.)
    - Normalizes symbol variations (& -> and)
    - Normalizes punctuation and removes noise symbols
    """
    if pd.isna(name_str):
        return ""
    
    text = str(name_str)
    
    # 1. Strip URLs / domain names
    text = URL_REGEX.sub('', text)
    
    # 2. Lowercase and NFKD unicode normalization
    text = unicodedata.normalize('NFKD', text).lower()
    
    # 3. Remove dots between letters (e.g., S.A.R.L. -> sarl, Inc. -> inc, P.V.T. -> pvt)
    text = re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b|\s|$)', '', text)
    
    # 4. Handle DBA / trade name markers
    text = DBA_REGEX.sub('', text)
    
    # 5. Replace '&' with 'and'
    text = re.sub(r'&', ' and ', text)
    
    # 6. Replace non-alphanumeric characters with single space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 7. Strip legal suffixes (multi-pass to catch combinations like 'pvt ltd co')
    for _ in range(2):
        text = LEGAL_SUFFIX_REGEX.sub('', text)
    
    # 8. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_address(addr_str: str) -> str:
    """
    Address parsing & standardization:
    - Converts to lower case & unicode normalization
    - Standardizes street variants (Rd -> road, St -> street, etc.)
    - Standardizes unit/number prefixes (H.No -> houseno, Apt -> apartment)
    """
    if pd.isna(addr_str):
        return ""
    
    text = str(addr_str)
    text = unicodedata.normalize('NFKD', text).lower()
    
    # Clean non-alphanumeric except spaces for abbreviation matching
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Standardize address abbreviations
    for pattern, repl in ADDR_REGEX_PATTERNS:
        text = pattern.sub(repl, text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_pincode(addr_str: str) -> Optional[str]:
    """Extract 5-digit (US/France) or 6-digit (India) postal/PIN code from address."""
    if pd.isna(addr_str):
        return None
    matches = PINCODE_REGEX.findall(str(addr_str))
    return matches[-1] if matches else None


def extract_tokens(text: str) -> List[str]:
    """Extract token list from clean text string."""
    return [t for t in text.split() if len(t) > 1]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the data preprocessing pipeline to a DataFrame with business records.
    Expects columns: entity_id, business_name, business_address, country
    Returns DataFrame with new clean columns:
    - clean_country
    - clean_business_name
    - clean_business_address
    - pincode
    - name_tokens
    - soundex_tokens
    """
    df = df.copy()
    df['clean_country'] = df['country'].apply(normalize_country)
    df['clean_business_name'] = df['business_name'].apply(normalize_business_name)
    df['clean_business_address'] = df['business_address'].apply(normalize_address)
    df['pincode'] = df['business_address'].apply(extract_pincode)
    df['name_tokens'] = df['clean_business_name'].apply(extract_tokens)
    df['soundex_tokens'] = df['clean_business_name'].apply(generate_soundex_tokens)
    return df


if __name__ == '__main__':
    sample_records = [
        {
            "entity_id": "S1-785847572",
            "business_name": "Consulting Nyasa Nursing Private Limited | www.nyasa.com",
            "business_address": "2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra 400078",
            "country": "India"
        },
        {
            "entity_id": "S2-764573417",
            "business_name": "-- Holloway Peak Inc Seafood dba Peak Foods",
            "business_address": "105 ELM ST, MORGANTON, NC 28655",
            "country": "US"
        },
        {
            "entity_id": "S3-10029381",
            "business_name": "Le Petit Bistro S.A.R.L.",
            "business_address": "12 Rue de la Paix, 75002 Paris",
            "country": "France"
        }
    ]
    sample_df = pd.DataFrame(sample_records)
    processed = preprocess_dataframe(sample_df)
    print("--- OPTIMIZED PREPROCESSING TEST ---")
    for _, row in processed.iterrows():
        print(f"ID: {row['entity_id']} ({row['clean_country']})")
        print(f"  Raw Name:     {row['business_name']}")
        print(f"  Clean Name:   {row['clean_business_name']}")
        print(f"  Name Tokens:  {row['name_tokens']}")
        print(f"  Soundex:      {row['soundex_tokens']}")
        print(f"  Clean Addr:   {row['clean_business_address']}")
        print(f"  PIN/Zip:      {row['pincode']}")
        print("-" * 50)
