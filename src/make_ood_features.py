"""Build ``ood_features.csv`` with the same 12 features used for training.

Run: ``python -m src.make_ood_features``
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
