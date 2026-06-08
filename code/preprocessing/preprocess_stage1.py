#!/usr/bin/env python3
"""
Stage 1 Preprocessing: Extract value/unit from catalog_content, normalize units.

Reads CSVs, extracts numeric value + unit from the last two lines of catalog_content,
normalizes units to {oz, fl oz, gram, ml, pound, count}, applies kg→g / L→mL / gal→mL
conversions, and falls back to regex or optional OpenAI LLM for missing values.
"""
import os, re, math, json
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# --- Optional OpenAI fallback ---
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception:
        USE_OPENAI = False

# --- Conversion tables (case-insensitive handled in code) ---
CONVERSIONS = {
    "kg": (1000, "gram"), "ltr": (1000, "ml"), "liter": (1000, "ml"),
    "liters": (1000, "ml"), "gallon": (3785, "ml"), "gallons": (3785, "ml"),
}

UNIT_MAP = {
    # Invalid / missing → NaN
    "-": np.nan, "---": np.nan, "": np.nan, "None": np.nan, "NA": np.nan,
    "....": np.nan, "product_weight": "gram", "Foot": np.nan, "in": np.nan,
    "M": np.nan, "Sq Ft": np.nan, "sq ft": np.nan, "(Pack of 1)": np.nan,
    # Ounce
    **dict.fromkeys(["oz", "ounce", "ounces", "OZ", "Ounce", "Ounces", "Oz", "per Box", "per Carton"], "oz"),
    # Fluid ounce
    **dict.fromkeys([
        "fl oz", "fl. oz.", "fl. oz", "Fl oz", "Fl Oz", "FL Oz", "FL OZ",
        "Fluid ounce", "Fluid Ounce", "Fluid Ounces", "fluid ounce",
        "fluid ounce(s)", "fluid ounces", "fluid_ounces", "Fl.oz", "Fl. OZ", "Fl OZ"
    ], "fl oz"),
    # Count
    **dict.fromkeys([
        "count", "Count", "COUNT", "ct", "CT", "Each", "each", "EA", "ea",
        "Piece", "Pieces", "Bag", "bag", "Box", "box", "Bottle", "bottle",
        "Pack", "PACK", "packs", "Packs", "Packet", "Pouch", "Carton",
        "Container", "Jar", "jar", "JARS", "K-Cups", "KIT", "Capsules",
        "Capsule", "Per Package", "Box/12", "BOX/12", "Bucket",
        "Paper Cupcake Liners", "Tea Bags", "tea bags", "unit", "units",
        "stück", "Cou", "unità", "Can", "can", "Tin", "Ziplock bags",
        "SACHET", "Sugar Substitute", "pac", "Count / Count", "bottles"
    ], "count"),
    # Grams
    **dict.fromkeys(["g", "gr", "gram", "grams", "Grams", "Gram", "Grams(gm)", "gramm"], "gram"),
    # Milliliter
    **dict.fromkeys([
        "ml", "mililitro", "millilitre", "milliliter", "millilitro",
        "Milliliters", "Liter", "Liters", "Ltr", "ltr", "liters"
    ], "ml"),
    # Pounds
    **dict.fromkeys(["lb", "LB", "Lbs", "lbs", "Pound", "Pounds", "pounds", "pound"], "pound"),
}

ALLOWED_UNITS = {"oz", "fl oz", "gram", "ml", "pound", "count"}

# --- Token-based fallback for unit lookup ---
TOKEN_MATCHES = {
    "fl oz": ["fl oz", "fl. oz", "fluid ounce", "fluid ounces", "fluid"],
    "oz": [" oz", "ounce", "ounces", "oz"],
    "gram": ["gram", "grams", "gr", " g"],
    "ml": ["ml", "milliliter", "millilitre", "liter", "liters", "ltr"],
    "pound": ["lb", "pound", "pounds", "lbs"],
    "count": ["pack", "box", "bottle", "bag", "each", "ea", "ct", "count",
              "piece", "pieces", "packet", "pouch", "jar", "capsule", "tin", "sachet"],
}


def safe_float(s) -> Optional[float]:
    """Robustly extract a float from a string."""
    if s is None:
        return None
    s = str(s).strip().replace("\u200b", "")
    if not s:
        return None
    if re.match(r"^\d+,\d+$", s):  # European decimal: "7,2" → "7.2"
        s = s.replace(",", ".")
    m = re.search(r"([-+]?\d*\.?\d+([eE][-+]?\d+)?)", s)
    return float(m.group(1)) if m else None


def extract_value_unit(catalog: str) -> Tuple[Optional[float], Optional[str], str]:
    """Parse value, unit, and description from catalog_content field."""
    if not isinstance(catalog, str):
        return None, None, ""
    lines = [ln.strip() for ln in catalog.strip().splitlines() if ln.strip()]
    unit = value = None

    if lines:
        last = lines[-1]
        unit = last.split(":", 1)[1].strip() if last.lower().startswith("unit:") else last.strip()
    if len(lines) >= 2:
        second_last = lines[-2]
        value = safe_float(second_last.split(":", 1)[1] if second_last.lower().startswith("value:") else second_last)

    desc = "\n".join(lines[:-2]).strip() if len(lines) > 2 else ""

    # Try extracting value from unit string (e.g. "1.76 Ounce")
    if value is None and unit:
        value = safe_float(unit)
        if value is not None:
            unit = re.sub(r"[-+]?\d*[,.]?\d+([eE][-+]?\d+)?", "", unit).strip() or None
    if value is None:
        value = safe_float(catalog)

    return value, unit, desc


def apply_conversion(unit: Optional[str], value: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    """Convert kg→gram, liter→ml, gallon→ml if applicable."""
    if unit is None or value is None:
        return value, unit
    key = unit.strip().lower()
    for k, (factor, target) in CONVERSIONS.items():
        if k.lower() == key:
            return value * factor, target
    return value, unit


def lookup_unit(raw: Optional[str]) -> Optional[str]:
    """Map raw unit string to one of ALLOWED_UNITS or NaN."""
    if not raw or not raw.strip():
        return np.nan
    ru = raw.strip()
    # Exact match
    if ru in UNIT_MAP:
        return UNIT_MAP[ru]
    # Case-insensitive
    for k, v in UNIT_MAP.items():
        if k.lower() == ru.lower():
            return v
    # Token search
    lru = ru.lower()
    for target, tokens in TOKEN_MATCHES.items():
        if any(t in lru for t in tokens):
            return target
    return np.nan


def heuristic_infer(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Regex patterns to find value+unit in free text."""
    if not text:
        return None, None
    txt = text.replace("\n", " ").lower()
    patterns = [
        (r"(\d+[,.]?\d*)\s*(ml|millilit(?:e|er)|millilitro)\b", "ml"),
        (r"(\d+[,.]?\d*)\s*(g|gram|grams|gr)\b", "gram"),
        (r"(\d+[,.]?\d*)\s*(kg|kilogram|kilograms)\b", "gram"),
        (r"(\d+[,.]?\d*)\s*(l|ltr|liter|liters)\b", "ml"),
        (r"(\d+[,.]?\d*)\s*(oz|ounce|ounces)\b", "oz"),
        (r"(\d+[,.]?\d*)\s*(fl\.?\s?oz|fluid ounce|fluid ounces)\b", "fl oz"),
        (r"(\d+[,.]?\d*)\s*(lb|pound|pounds|lbs)\b", "pound"),
        (r"(?:pack of|pack)\s*(\d+)", "count"),
        (r"(\d+)\s*(?:count|ct|pcs|pieces)\b", "count"),
        (r"(\d+)\s*(?:bottle|bottles|box|boxes|bag|bags|pouch|jar|jars)\b", "count"),
    ]
    for pat, unit in patterns:
        m = re.search(pat, txt)
        if m:
            num = safe_float(m.group(1))
            if num is not None:
                if unit == "gram" and "kg" in m.group(0):
                    num *= 1000
                if unit == "ml" and re.search(r"\bl\b|\bltr\b|\bliter\b", m.group(0)):
                    num *= 1000
                return float(num), unit
    return None, None


def infer_with_llm(description: str) -> Tuple[float, str]:
    """Try heuristic first, then optional OpenAI fallback. Default: (1.0, 'count')."""
    val, unit = heuristic_infer(description or "")
    if unit:
        return val, unit

    if USE_OPENAI:
        try:
            prompt = (
                "Given this product description, infer numeric value and unit from "
                "[oz, fl oz, gram, ml, pound, count]. Return JSON: {\"value\": <num>, \"unit\": \"<unit>\"}.\n\n"
                f"Description:\n{description}\n\n"
                "If unknown, return {\"value\": 1, \"unit\": \"count\"}."
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                max_tokens=60, temperature=0.0,
            )
            obj = json.loads(re.search(r"\{.*\}", resp["choices"][0]["message"]["content"], re.S).group(0))
            v = safe_float(obj.get("value")) or 1.0
            u = obj.get("unit", "count").strip().lower()
            u = {"grams": "gram", "g": "gram", "milliliter": "ml", "millilitre": "ml"}.get(u, u)
            return float(v), u if u in ALLOWED_UNITS else "count"
        except Exception:
            pass

    return 1.0, "count"


def process_dataframe(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Add normalized value/unit columns with LLM/heuristic fallback."""
    records = []
    for idx, row in df.iterrows():
        raw_val, raw_unit, desc = extract_value_unit(row.get("catalog_content", ""))
        val, unit = apply_conversion(raw_unit, raw_val)
        records.append({"_idx": idx, "raw_value": raw_val, "raw_unit": raw_unit,
                         "desc_for_llm": desc, "value_after_conv": val, "unit_after_conv": unit})

    meta = pd.DataFrame(records).set_index("_idx")
    out = df.copy().join(meta, how="left")

    # Cap unreasonable values
    big = out["value_after_conv"].apply(lambda x: isinstance(x, (int, float)) and not math.isnan(x) and x > 10000)
    print(f"[{name}] {int(big.sum())} rows with value > 10000 → set to NaN")
    out.loc[big, ["value_after_conv", "unit_after_conv"]] = [np.nan, None]

    # Normalize units
    out["unit_normalized"] = out["unit_after_conv"].apply(lookup_unit)

    # Infer missing via heuristic/LLM
    missing = out["unit_normalized"].isna()
    print(f"[{name}] {int(missing.sum())} rows need heuristic/LLM inference")
    for idx in out[missing].index:
        v, u = infer_with_llm(out.at[idx, "desc_for_llm"] or "")
        out.at[idx, "value_after_conv"] = float(v)
        out.at[idx, "unit_after_conv"] = u
        out.at[idx, "unit_normalized"] = u if u in ALLOWED_UNITS else "count"

    # Final validation
    out["unit_normalized"] = out["unit_normalized"].apply(
        lambda u: u if isinstance(u, str) and u in ALLOWED_UNITS else np.nan
    )
    print(f"[{name}] Unit distribution:\n{out['unit_normalized'].value_counts(dropna=False)}\n")
    return out


def main():
    files = [
        ("/kaggle/input/amazon-ml-dataset-csv/splits/splits/train.csv", "train_split_norm.csv"),
        ("/kaggle/input/amazon-ml-dataset-csv/splits/splits/val.csv", "val_split_norm.csv"),
        ("/kaggle/input/amazon-ml-dataset-csv/dataset/dataset/test.csv", "test_norm.csv"),
    ]
    for inp, outp in files:
        if not os.path.exists(inp):
            print(f"Skipping {inp} (not found)")
            continue
        print(f"\nProcessing {inp}...")
        df = pd.read_csv(inp)
        result = process_dataframe(df, name=inp)
        result.to_csv(outp, index=False)
        print(f"Saved → {outp}")


if __name__ == "__main__":
    main()
