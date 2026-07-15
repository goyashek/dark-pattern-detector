"""Train the classical dark-pattern classifier with source-aware validation.

The model combines character TF-IDF with 12 lexical/structural features, SMOTE and
LinearSVC. Model selection uses connected page/template groups on the training
partition only. The selected classifier is then sigmoid-calibrated on grouped folds.

Run: python -m src.train
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features import NUM_COLS as MODEL_NUM_COLS
from src.leak_audit import connected_groups, dataset_hash

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH = os.path.join(HERE, "data", "processed", "features.csv")
OOD_PATH = os.path.join(HERE, "data", "processed", "ood_features.csv")
SPLIT_PATH = os.path.join(HERE, "reports", "leak_free_split.json")
MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")
RANDOM_STATE = 42
def load_leakfree_split(df):
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit(f"Missing {SPLIT_PATH}; run `python -m src.leak_audit` first.")

    with open(SPLIT_PATH) as fh:
        split = json.load(fh)
    train = np.asarray(split["train"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    expected = set(range(len(df)))

    if set(train) & set(test) or set(train) | set(test) != expected:
        raise SystemExit("Invalid split: indices must be unique, disjoint, and cover every row.")
    if split.get("dataset_sha256") != dataset_hash(df):
        raise SystemExit("Stale split: dataset hash differs; rerun `python -m src.leak_audit`.")

    groups = connected_groups(df)
    if set(groups[train]) & set(groups[test]):
        raise SystemExit("Leaky split: a page/template group appears in both train and test.")
    return train, test, groups


def make_model(c=1.0):
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 6), min_df=2,
            max_features=30_000, sublinear_tf=True,
        ), "text"),
        ("numeric", Pipeline([
            ("scale", RobustScaler()),
            ("power", PowerTransformer(method="yeo-johnson")),
        ]), MODEL_NUM_COLS),
    ])
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LinearSVC(
            C=c, random_state=RANDOM_STATE, max_iter=2_000, tol=1e-3,
        )),
    ])


def grouped_folds(text, labels, groups, n_splits=5):
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE,
    )
    folds = list(cv.split(text, labels, groups))
    classes = set(labels)
    if any(set(labels[train]) != classes or set(labels[test]) != classes for train, test in folds):
        raise SystemExit("A grouped CV fold is missing a class; revise the grouping or fold count.")
    return folds


def calibrated_model(model, folds):
    return CalibratedClassifierCV(model, method="sigmoid", cv=folds, ensemble=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(FEAT_PATH)
    df["text"] = df["text"].fillna("")
    data = df[["text"] + MODEL_NUM_COLS]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["Pattern Category"])
    train_idx, test_idx, groups = load_leakfree_split(df)

    x_train, x_test = data.iloc[train_idx], data.iloc[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    train_groups = groups[train_idx]
    folds = grouped_folds(x_train, y_train, train_groups)
    print(f"Loaded {len(df)} rows: {len(train_idx)} train / {len(test_idx)} source-clean test")

    rows = []
    for c in (0.5, 1.0, 2.0):
        scores = cross_val_score(
            make_model(c), x_train, y_train,
            cv=folds, scoring="f1_macro", n_jobs=-1,
        )
        row = {
            "C": c,
            "CV Macro-F1": float(scores.mean()),
            "CV Std": float(scores.std()),
        }
        rows.append(row)
        print(f"C={c:<3}: {scores.mean():.4f} +/- {scores.std():.4f}")

    comparison = pd.DataFrame(rows).sort_values(
        ["CV Macro-F1", "CV Std"], ascending=[False, True]
    ).reset_index(drop=True)
    comparison.to_csv(os.path.join(REPORTS_DIR, "cv_model_comparison.csv"), index=False)
    winner = comparison.iloc[0]
    params = {
        "c": float(winner["C"]),
    }

    calibration_folds = grouped_folds(x_train, y_train, train_groups, n_splits=3)
    selected = calibrated_model(make_model(**params), calibration_folds)
    selected.fit(x_train, y_train)
    predictions = selected.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    print(f"Held-out test: accuracy={test_accuracy:.4f}, macro-F1={test_f1:.4f}")

    report = classification_report(
        y_test, predictions, target_names=encoder.classes_, zero_division=0
    )
    with open(os.path.join(REPORTS_DIR, "test_classification_report.txt"), "w") as fh:
        fh.write(f"Held-out test accuracy: {test_accuracy:.4f}\n")
        fh.write(f"Held-out test macro-F1: {test_f1:.4f}\n\n{report}")

    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(encoder.classes_)))
    ax.set_yticks(range(len(encoder.classes_)))
    ax.set_xticklabels(encoder.classes_, rotation=90, fontsize=7)
    ax.set_yticklabels(encoder.classes_, fontsize=7)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Character LinearSVC (test macro-F1={test_f1:.3f})")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    ood_f1 = None
    if os.path.exists(OOD_PATH):
        ood = pd.read_csv(OOD_PATH)
        known = ood["Pattern Category"].isin(encoder.classes_)
        ood_y = encoder.transform(ood.loc[known, "Pattern Category"])
        ood_pred = selected.predict(ood.loc[known, ["text"] + MODEL_NUM_COLS])
        ood_f1 = f1_score(ood_y, ood_pred, average="macro", zero_division=0)
        print(f"OOD development set: macro-F1={ood_f1:.4f} ({known.sum()} rows)")

    # Refit the selected, calibrated pipeline on all rows for deployment.
    deployment_folds = grouped_folds(data, labels, groups, n_splits=3)
    deployment = calibrated_model(make_model(**params), deployment_folds)
    deployment.fit(data, labels)
    joblib.dump(deployment, os.path.join(MODELS_DIR, "best_multi_model.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    metrics = {
        "model": "character TF-IDF + 12 engineered features + SMOTE + calibrated LinearSVC",
        "n_rows": int(len(df)),
        "n_classes": int(len(encoder.classes_)),
        "split_grouping": "connected_page_or_skeleton_v1",
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_f1),
        "ood_dev_macro_f1": None if ood_f1 is None else float(ood_f1),
        "engineered_features": MODEL_NUM_COLS,
        "best_params": {"C": params["c"], "ngram_range": [2, 6]},
        "best_cv_macro_f1": float(winner["CV Macro-F1"]),
        "best_cv_std": float(winner["CV Std"]),
        "cv_comparison": comparison.to_dict(orient="records"),
    }
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved calibrated model, label encoder, and reports.")


if __name__ == "__main__":
    main()
