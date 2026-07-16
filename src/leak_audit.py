"""
leak_audit.py — Template-sibling leak audit + template-aware honest split.

WHY THIS EXISTS
---------------
The training pool is heavily templated: each row is a fixed sentence skeleton with a
few slots (brand, product, price, fee...) filled in. A plain random train/test split
can put one sibling of a template in train and another sibling of the SAME template in
test. The two strings differ only by a brand or a price, so any model effectively sees
the test answer at fit time — the reported macro-F1 is inflated.

This script:
  1. Reconstructs each row's TEMPLATE SKELETON by masking every generator slot value
     (currency, numbers, and the exact brand/product/fee/etc. vocab from collect_data.py),
     then clusters rows by identical skeleton.
  2. Quantifies leakage in the naive split NB2 uses
     (train_test_split, test_size=0.2, stratify=y, random_state=42): how many skeleton
     clusters straddle train and test, and how many test rows have a template twin in train.
  3. Builds a SOURCE-AWARE split. Rows are connected when they share either a template
     skeleton or page_id, then StratifiedGroupKFold keeps each connected component intact.
  4. Trains the legacy notebook-2 pipelines on both splits to retain the original leakage
     comparison. The current character-SVC evaluation lives in src/train.py.
  5. Runs trivial single-signal probes (has-currency-symbol, text-length, keyword-only)
     to check whether a lone confound can predict the class.

Outputs (under reports/):
  - leak_audit.json        full report (counts, cluster stats, naive-vs-leak-free F1, probes)
  - leak_free_split.json   row-index lists {train, test} for the template-aware split

Exit code is NON-ZERO if the leak-free split is not actually clean (a template still
straddles the two sides) — so this can gate downstream training.

Run:  python -m src.leak_audit
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Slot vocabularies — imported from the generator so the skeleton mask matches exactly
# what was filled in. Keeping this as an import (not a copy) means the audit stays correct
# if the pools ever change.
from src.collect_data import (
    ADDONS, BRANDS, CATEGORIES, DURATIONS, FEES, FILE_TYPES, PRODUCTS, SOFTWARE,
)
from src.features import NUM_COLS

SEED = 42
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(HERE, "data", "processed", "features.csv")
REPORTS = os.path.join(HERE, "reports")

# Tuned hyperparameters lifted straight from notebook 2's Optuna study (the params that
# produced the deployed joblib models), so the inflation figure is for the REAL deployed
# model, not a proxy.
XGB_BEST = dict(n_estimators=105, max_depth=9, learning_rate=0.20396887264773383,
                subsample=0.7637017332034828, colsample_bytree=0.7545474901621302)
SVC_BEST = dict(C=2.124280213720889, loss="squared_hinge")


# --------------------------------------------------------------------------- #
# 1. Template skeleton
# --------------------------------------------------------------------------- #
def _vocab_pattern():
    """One big alternation matching every slot value, longest-first so multi-word
    names (e.g. 'Protect Promise Fee') mask before their substrings."""
    vocab = set()
    for pool in (BRANDS, PRODUCTS, FEES, SOFTWARE, FILE_TYPES, ADDONS, CATEGORIES, DURATIONS):
        for v in pool:
            vocab.add(v.lower())
            vocab.add(v)  # original case too
    terms = sorted(vocab, key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)


_VOCAB_RE = _vocab_pattern()
_CUR_RE = re.compile(r"[₹$]\s?\d[\d,]*(?:\.\d+)?")   # ₹1,499.00 / $99.00
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")           # any remaining number
_PCT_RE = re.compile(r"\d+\s?%")
_WS_RE = re.compile(r"\s+")


def skeleton(text: str) -> str:
    """Collapse a row to its template skeleton: mask currency, percentages, numbers,
    and all known slot vocab, then normalise whitespace/case. Two rows generated from
    the same template (differing only by filled slots) yield the SAME skeleton."""
    s = str(text)
    s = _CUR_RE.sub("<CUR>", s)
    s = _PCT_RE.sub("<PCT>", s)
    s = _VOCAB_RE.sub("<SLOT>", s)
    s = _NUM_RE.sub("<NUM>", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def connected_groups(df: pd.DataFrame) -> np.ndarray:
    """Group rows transitively by either source page or generated-text skeleton."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    page_ids = df["page_id"] if "page_id" in df else pd.Series(index=df.index, dtype=object)
    for values in (page_ids, df["text"].fillna("").map(skeleton)):
        seen = {}
        for i, value in enumerate(values):
            if pd.isna(value) or str(value).strip() == "":
                continue
            key = str(value)
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    return pd.factorize(np.asarray([find(i) for i in range(len(df))]))[0]


def dataset_hash(df: pd.DataFrame) -> str:
    """Hash the row order and fields that define a saved positional split."""
    cols = ["page_id", "text", "label", "Pattern Category"]
    payload = df[cols].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# 2. Pipelines mirroring notebook 2
# --------------------------------------------------------------------------- #
def _preprocessor():
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=300, ngram_range=(1, 3), min_df=2,
                                 sublinear_tf=True), "clean_text"),
        ("num", Pipeline([("scale", RobustScaler()),
                          ("power", PowerTransformer(method="yeo-johnson"))]), NUM_COLS),
    ])


def _svc_pipe():
    return ImbPipeline([
        ("prep", _preprocessor()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", LinearSVC(**SVC_BEST, class_weight="balanced", random_state=SEED,
                          max_iter=5000)),
    ])


def _xgb_pipe():
    return ImbPipeline([
        ("prep", _preprocessor()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", XGBClassifier(**XGB_BEST, eval_metric="mlogloss", random_state=SEED,
                              tree_method="hist")),
    ])


_MODELS = {"legacy_svc": _svc_pipe, "legacy_xgb": _xgb_pipe}


def _macro_f1(pipe_fn, X_tr, y_tr, X_te, y_te):
    pipe = pipe_fn()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return float(f1_score(y_te, pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# 3. Splits
# --------------------------------------------------------------------------- #
def naive_split(n, y):
    """The split notebook 2 uses: stratified random 80/20, seed 42."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    return tr, te


def template_aware_split(y, groups):
    """80/20 split where no skeleton cluster straddles the two sides, while keeping
    class proportions as close as a group constraint allows.

    We take the FIRST fold of a 5-split StratifiedGroupKFold whose test side contains
    every class. Blindly taking fold 0 (the old behaviour) can leave a class with a few
    large, indivisible skeleton clusters entirely on the train side — e.g. 'Bait and
    Switch', whose rows are all unique skeletons, landed 320-in-train / 0-in-test, making
    it unevaluable and dragging macro-F1 down by a phantom F1=0. Every fold is still
    leak-free by construction (StratifiedGroupKFold never splits a group), so scanning
    for a fully-covered fold costs nothing and only fixes coverage."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_classes = len(np.unique(y))
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        if len(np.unique(y[te])) == n_classes:
            return tr, te
    # No fold covers all classes (a class has too few skeleton clusters to ever land in
    # test). Fall back to the most-covered fold and let main()'s report surface the gap.
    tr, te = max(folds, key=lambda f: len(np.unique(y[f[1]])))
    return tr, te


def cross_split_clusters(groups, tr, te):
    """Skeleton clusters that appear on BOTH sides of a split (the leakage)."""
    g = np.asarray(groups)
    g_tr, g_te = set(g[tr]), set(g[te])
    shared = g_tr & g_te
    # how many test rows have a twin in train
    leaked_test_rows = int(np.isin(g[te], list(shared)).sum())
    return shared, leaked_test_rows


# --------------------------------------------------------------------------- #
# 4. Trivial probes
# --------------------------------------------------------------------------- #
def _probe_f1(feat_2d, y, tr, te):
    """Train logistic regression on a single (or tiny) feature matrix; report macro-F1."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(feat_2d[tr], y[tr])
    pred = clf.predict(feat_2d[te])
    return float(f1_score(y[te], pred, average="macro", zero_division=0))


def run_probes(df, y, tr, te):
    probes = {}

    # (a) currency-symbol probe: does "has $" / "has ₹" alone predict a class?
    has_dollar = df["text"].str.contains(r"\$", regex=True, na=False).astype(int)
    has_rupee = df["text"].str.contains("₹", na=False).astype(int)
    cur = np.c_[has_dollar.to_numpy(), has_rupee.to_numpy()]
    probes["currency_symbol"] = {
        "macro_f1": _probe_f1(cur, y, tr, te),
        "n_dollar_rows": int(has_dollar.sum()),
        "n_rupee_rows": int(has_rupee.sum()),
    }

    # (b) text-length probe
    length = df["text"].fillna("").str.len().to_numpy().reshape(-1, 1).astype(float)
    probes["text_length"] = {"macro_f1": _probe_f1(length, y, tr, te)}

    # (c) keyword-only probe: TF-IDF on clean_text, no numeric features, no SMOTE
    kw = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 3), min_df=2,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED)),
    ])
    kw.fit(df["clean_text"].iloc[tr], y[tr])
    kw_pred = kw.predict(df["clean_text"].iloc[te])
    probes["keyword_only"] = {"macro_f1": float(f1_score(y[te], kw_pred,
                                                         average="macro", zero_division=0))}
    return probes


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(FEATURES)
    df["clean_text"] = df["clean_text"].fillna("")
    n = len(df)

    # skeleton and source-page clustering
    df["skeleton"] = df["text"].map(skeleton)
    skel_codes = df["skeleton"].astype("category").cat.codes.to_numpy()
    group_codes = connected_groups(df)
    n_clusters = int(df["skeleton"].nunique())
    sib_clusters = int((df["skeleton"].value_counts() > 1).sum())
    largest = df["skeleton"].value_counts().head(5)

    le = LabelEncoder()
    y = le.fit_transform(df["Pattern Category"])
    X = df[["clean_text"] + NUM_COLS]

    # naive split + its leakage
    tr_n, te_n = naive_split(n, y)
    shared_n, leaked_rows_n = cross_split_clusters(skel_codes, tr_n, te_n)
    shared_source_n, leaked_source_rows_n = cross_split_clusters(group_codes, tr_n, te_n)

    # source-aware split + confirm neither pages nor templates straddle it
    tr_g, te_g = template_aware_split(y, group_codes)
    shared_g, leaked_rows_g = cross_split_clusters(group_codes, tr_g, te_g)
    shared_skeleton_g, _ = cross_split_clusters(skel_codes, tr_g, te_g)

    # Historical macro-F1 under each split for the old notebook-2 pipelines.
    per_model = {}
    for key, fn in _MODELS.items():
        f1_naive = _macro_f1(fn, X.iloc[tr_n], y[tr_n], X.iloc[te_n], y[te_n])
        f1_leakfree = _macro_f1(fn, X.iloc[tr_g], y[tr_g], X.iloc[te_g], y[te_g])
        per_model[key] = {
            "macro_f1_naive": round(f1_naive, 4),
            "macro_f1_leakfree": round(f1_leakfree, 4),
            "macro_f1_drop": round(f1_naive - f1_leakfree, 4),
            "relative_pct": round(100 * (f1_naive - f1_leakfree) / f1_naive, 1) if f1_naive else None,
        }

    # trivial probes (run on the leak-free split so confounds aren't themselves inflated)
    probes = run_probes(df, y, tr_g, te_g)

    report = {
        "n_rows": n,
        "n_classes": int(df["Pattern Category"].nunique()),
        "skeleton_clusters": {
            "total_clusters": n_clusters,
            "multi_row_clusters": sib_clusters,
            "largest_clusters": {k: int(v) for k, v in largest.items()},
        },
        "connected_source_groups": int(len(np.unique(group_codes))),
        "naive_split": {
            "test_size": int(len(te_n)),
            "cross_split_clusters": len(shared_n),
            "leaked_test_rows": leaked_rows_n,
            "leaked_test_pct": round(100 * leaked_rows_n / len(te_n), 1),
            "cross_split_source_groups": len(shared_source_n),
            "source_leaked_test_rows": leaked_source_rows_n,
        },
        "leak_free_split": {
            "train_size": int(len(tr_g)),
            "test_size": int(len(te_g)),
            "cross_split_clusters": len(shared_g),
            "leaked_test_rows": leaked_rows_g,
            "cross_split_skeletons": len(shared_skeleton_g),
        },
        "models": per_model,
        "headline_model": "legacy_xgb",
        "trivial_probes": probes,
        "pipeline": "legacy TF-IDF family + the current 12 numeric features + SMOTE; current model is evaluated by src/train.py",
        "seed": SEED,
    }

    with open(os.path.join(REPORTS, "leak_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(REPORTS, "leak_free_split.json"), "w") as f:
        json.dump({
            "train": tr_g.tolist(),
            "test": te_g.tolist(),
            "seed": SEED,
            "grouping": "connected_page_or_skeleton_v1",
            "dataset_sha256": dataset_hash(df),
        }, f, indent=2)

    # console summary
    print("=" * 64)
    print("LEAK AUDIT")
    print("=" * 64)
    print(f"rows={n}  classes={report['n_classes']}  "
          f"skeleton clusters={n_clusters}  (multi-row={sib_clusters})")
    print("top skeletons:")
    for k, v in largest.items():
        print(f"   {v:4d}  {k[:80]}")
    print("-" * 64)
    print(f"NAIVE split (NB2):     test={len(te_n)}  "
          f"cross-split templates={len(shared_n)}  "
          f"leaked test rows={leaked_rows_n} ({report['naive_split']['leaked_test_pct']}%)")
    print(f"LEAK-FREE split:       test={len(te_g)}  "
          f"cross-split source groups={len(shared_g)}  leaked test rows={leaked_rows_g}")
    print("-" * 64)
    print(f"{'model':6s} {'naive':>8s} {'leak-free':>10s} {'drop':>8s} {'rel%':>7s}")
    for key, m in per_model.items():
        print(f"{key:6s} {m['macro_f1_naive']:8.4f} {m['macro_f1_leakfree']:10.4f} "
              f"{m['macro_f1_drop']:8.4f} {m['relative_pct']:6.1f}%")
    print("-" * 64)
    print("trivial probes (macro-F1 on leak-free test):")
    for name, p in probes.items():
        print(f"   {name:18s} {p['macro_f1']:.4f}")
    print("=" * 64)
    print(f"wrote {os.path.join('reports', 'leak_audit.json')}")
    print(f"wrote {os.path.join('reports', 'leak_free_split.json')}")

    # gate: the leak-free split MUST be clean
    if shared_g or shared_skeleton_g:
        print(f"\nFAIL: leak-free split still has {len(shared_g)} source groups and "
              f"{len(shared_skeleton_g)} skeletons crossing the split",
              file=sys.stderr)
        return 1
    print("\nOK: leak-free split is page- and template-clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
