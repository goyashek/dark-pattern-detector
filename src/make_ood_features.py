"""make_ood_features.py — Precompute the 12 features for the OOD real-world test rows.

Reads  data/processed/ood_real_test.csv   (raw real Indian UI strings + labels)
Writes data/processed/ood_features.csv    (text + clean_text + 12 numeric features + labels)

Uses the SAME shared extractor as training (src/features.py), so the classical model sees
identical inputs on OOD as it did in training — no train-serve skew. This is done here (not
in the notebook) so notebook 3 only ever *loads* files and never recomputes features.

Run:  python -m src.make_ood_features
"""

import os
import pandas as pd

from src import features as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(HERE, "data", "processed", "ood_real_test.csv")
OUT_PATH = os.path.join(HERE, "data", "processed", "ood_features.csv")


def main():
    F.ensure_nltk()
    df = pd.read_csv(IN_PATH)
    df = df.dropna(subset=["text", "Pattern Category"]).reset_index(drop=True)
    print(f"Extracting features for {len(df)} OOD rows...")

    clean = [F.clean_and_lemmatize(t) for t in df["text"].astype(str)]
    feats = pd.DataFrame([F.extract_features(t) for t in df["text"].astype(str)])[F.NUM_COLS]

    keep = [c for c in ["text", "Pattern Category", "source", "url"] if c in df.columns]
    out = pd.concat(
        [df[keep].reset_index(drop=True),
         pd.Series(clean, name="clean_text"),
         feats.reset_index(drop=True)],
        axis=1,
    )
    out["clean_text"] = out["clean_text"].fillna("")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
