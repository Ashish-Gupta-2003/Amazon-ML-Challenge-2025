"""
Stage 2 Preprocessing: Convert all units to oz/fl_oz base + extract pack Count.

Input:  Stage 1 output CSV (with value_after_conv, unit_normalized, desc_for_llm)
Output: CSV with three engineered features: Count, oz, fl_oz
"""
import re
import pandas as pd
import numpy as np

# --- Conversion to base units ---
GRAMS_PER_OZ = 28.3495
ML_PER_FL_OZ = 29.5735

UNIT_TO_BASE = {
    'oz':    ('oz',    1.0),
    'gram':  ('oz',    1.0 / GRAMS_PER_OZ),
    'pound': ('oz',    16.0),
    'fl oz': ('fl_oz', 1.0),
    'ml':    ('fl_oz', 1.0 / ML_PER_FL_OZ),
    'count': ('count', 1.0),
}

# --- Pack count extraction ---
WORDS_TO_NUM = {w: i for i, w in enumerate(
    ['zero','one','two','three','four','five','six','seven','eight','nine',
     'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
     'seventeen','eighteen','nineteen','twenty']
)}

WORD_PATTERN = '|'.join(WORDS_TO_NUM.keys())

PACK_PATTERNS = [
    r'(\d+)\s*-?\s*pack\b',
    r'pack(?:\s+of)?\s+(\d+)\b',
    r'(\d+)\s*pk\b',
    r'(\d+)\s*[xX]\s*pack',
    rf'({WORD_PATTERN})\s*-?\s*pack\b',
    rf'pack(?:\s+of)?\s+({WORD_PATTERN})\b',
    r'(\d+)\s*(?:count|ct|cts)\b',
]


def extract_pack_count(desc: str) -> int | None:
    """Extract pack count from description text. Returns int or None."""
    if not isinstance(desc, str):
        return None
    s = desc.lower()
    for pat in PACK_PATTERNS:
        m = re.search(pat, s)
        if m:
            token = m.group(1)
            if token in WORDS_TO_NUM:
                return WORDS_TO_NUM[token]
            try:
                val = int(token)
                return val if val > 0 else None
            except ValueError:
                continue
    return None


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert units to oz/fl_oz and extract pack counts."""
    required = {'desc_for_llm', 'value_after_conv', 'unit_normalized'}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    units = df['unit_normalized'].astype(str).str.strip().str.lower()
    values = pd.to_numeric(df['value_after_conv'], errors='coerce')

    df['Count'] = np.nan
    df['oz'] = 0.0
    df['fl_oz'] = 0.0

    for idx in df.index:
        unit, val = units[idx], values[idx]
        if unit == 'count':
            df.at[idx, 'Count'] = val if not pd.isna(val) else np.nan
            continue

        if pd.isna(val) or unit not in UNIT_TO_BASE:
            continue

        base, factor = UNIT_TO_BASE[unit]
        if base == 'oz':
            df.at[idx, 'oz'] = float(val) * factor
        elif base == 'fl_oz':
            df.at[idx, 'fl_oz'] = float(val) * factor

        # Extract pack count for non-count units
        pack = extract_pack_count(df.at[idx, 'desc_for_llm'] if 'desc_for_llm' in df.columns else '')
        df.at[idx, 'Count'] = pack if pack else 1

    df['Count'] = df['Count'].fillna(0).astype(int)
    return df


def main(input_csv: str, output_csv: str = None):
    df = pd.read_csv(input_csv)
    out = process_dataframe(df)
    output_csv = output_csv or input_csv.rsplit('.', 1)[0] + '_final.csv'
    out.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")


if __name__ == '__main__':
    # Configure paths here
    main('/kaggle/input/amazon-ml-dataset-csv/preprocessed/test_norm.csv',
         '/kaggle/working/test_split_final.csv')
