"""
make_features.py — Apply the shared feature extractor to the assembled dataset.

Reads  data/processed/ccpa_dataset.tsv
Writes data/processed/features.csv  (text + clean_text + 12 numeric features + labels)

Run:  python -m src.make_features
"""

import os
import pandas as pd

from src import features as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(HERE, "data", "processed", "ccpa_dataset.tsv")
OUT_PATH = os.path.join(HERE, "data", "processed", "features.csv")


def main():
    F.ensure_nltk()
    df = pd.read_csv(IN_PATH, sep="\t", on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["text", "Pattern Category"]).reset_index(drop=True)

    print(f"Extracting 12 engineered features for {len(df)} rows...")
    clean, feats = [], []
    for i, txt in enumerate(df["text"].astype(str)):
        clean.append(F.clean_and_lemmatize(txt))
        feats.append(F.extract_features(txt))
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(df)}")

    feat_df = pd.DataFrame(feats)[F.NUM_COLS]
    out = pd.concat(
        [df[["page_id", "text", "label", "Pattern Category"]].reset_index(drop=True),
         pd.Series(clean, name="clean_text"),
         feat_df.reset_index(drop=True)],
        axis=1,
    )
    out["clean_text"] = out["clean_text"].fillna("")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]} cols -> {OUT_PATH}")
    print(out[["text", "clean_text", "Pattern Category"]].head(3).to_string())


if __name__ == "__main__":
    main()
