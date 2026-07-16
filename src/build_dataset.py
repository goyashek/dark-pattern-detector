"""Map and merge the Kaggle source tables into the final dataset.

The script removes normalized duplicate text before writing
``data/processed/ccpa_dataset.tsv``.

Run: ``python -m src.build_dataset``
"""

import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(HERE, "data", "raw", "dataset_raw.tsv")
COLLECTED_PATH = os.path.join(HERE, "data", "processed", "collected.tsv")
OUT_PATH = os.path.join(HERE, "data", "processed", "ccpa_dataset.tsv")


def map_to_ccpa(cat, text):
    """Map a Mathur/academic category + its text to a CCPA category."""
    text = str(text).lower()
    if cat == "Not Dark Pattern":
        return "Not a Dark Pattern"
    if cat in ("Urgency", "Scarcity"):
        return "False Urgency"
    if cat == "Social Proof":
        return "Disguised Advertisement"
    if cat == "Misdirection":
        if any(p in text for p in ["no thanks", "i don't want", "prefer to pay",
                                   "hate saving", "i prefer"]):
            return "Confirm Shaming"
        if any(p in text for p in ["?", "yes,", "no,", "opt-in", "uncheck", "pre-checked"]):
            return "Trick Question"
        return "Interface Interference"
    if cat == "Obstruction":
        if any(p in text for p in ["cancel", "membership", "subscription", "renew", "bill",
                                   "fee", "hotline"]):
            return "Subscription Trap"
        return "Interface Interference"
    if cat == "Sneaking":
        if any(p in text for p in ["fee", "charge", "tax", "cost", "surcharge",
                                   "processing", "booking"]):
            return "Drip Pricing"
        return "Basket Sneaking"
    if cat == "Forced Action":
        return "Forced Action"
    return "Not a Dark Pattern"


def normalise(text):
    """Lowercase, collapse whitespace, strip punctuation for dedup key only."""
    t = str(text).lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_raw():
    # on_bad_lines='skip' drops the handful of rows with embedded tabs/newlines
    # that broke the original parse.
    df = pd.read_csv(RAW_PATH, sep="\t", on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["text", "Pattern Category"])
    df["Pattern Category"] = df.apply(
        lambda r: map_to_ccpa(r["Pattern Category"], r["text"]), axis=1)
    # label: 1 for any dark pattern, 0 for benign
    df["label"] = (df["Pattern Category"] != "Not a Dark Pattern").astype(int)
    return df[["page_id", "text", "label", "Pattern Category"]]


def load_collected():
    df = pd.read_csv(COLLECTED_PATH, sep="\t", on_bad_lines="skip", engine="python")
    return df[["page_id", "text", "label", "Pattern Category"]]


def main():
    raw = load_raw()
    collected = load_collected()
    print(f"Raw (post-remap):  {len(raw)} rows")
    print(f"Collected:         {len(collected)} rows")

    combined = pd.concat([raw, collected], ignore_index=True)

    # --- drop degenerate rows ---------------------------------------------
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() >= 3]
    combined = combined.dropna(subset=["text", "Pattern Category", "label"])

    # --- GLOBAL DEDUP (the leakage fix) -----------------------------------
    combined["_key"] = combined["text"].apply(normalise)
    before = len(combined)
    combined = combined.drop_duplicates(subset="_key", keep="first")
    after = len(combined)
    combined = combined.drop(columns="_key")
    print(f"Removed {before - after} exact/near-duplicate strings (kept {after}).")

    combined["label"] = combined["label"].astype(int)
    combined = combined.reset_index(drop=True)
    combined.to_csv(OUT_PATH, sep="\t", index=False)

    print(f"\nFinal dataset: {len(combined)} unique rows -> {OUT_PATH}")
    print("\nClass distribution:")
    print(combined["Pattern Category"].value_counts().to_string())
    print("\nBinary label distribution:")
    print(combined["label"].value_counts().to_string())
    # sanity: confirm no remaining duplicate texts
    assert combined["text"].apply(normalise).duplicated().sum() == 0, "duplicates remain!"
    print("\n[OK] No duplicate strings remain in the final dataset.")


if __name__ == "__main__":
    main()
