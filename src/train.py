"""
train.py — Leakage-safe training, tuning and honest evaluation (Route 1: classical ML).

Methodology fixes vs. the original project:
  * GLOBAL DEDUP already done upstream, so no string appears in both train and test.
  * A held-out TEST set (20%) is carved out FIRST and never touched during tuning.
  * 5 classical models compared with Stratified 5-fold CV; SMOTE is fit INSIDE each
    fold (via imblearn Pipeline) so no synthetic minority sample leaks into validation.
  * Optuna tunes XGBoost using CROSS-VALIDATION macro-F1 on the training split only
    (the original tuned against the test set -> optimistic + leaky).
  * Final tuned model is evaluated ONCE on the untouched test set; we save the
    classification report, confusion matrix and model-comparison table.

Run:  python -m src.train
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH = os.path.join(HERE, "data", "processed", "features.csv")
MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")
RANDOM_STATE = 42


def make_preprocessor(scaler):
    return ColumnTransformer(transformers=[
        ("text_tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2),
                                       min_df=2, sublinear_tf=True), "clean_text"),
        ("num", scaler, F.NUM_COLS),
    ])


def std_numeric():
    return Pipeline([("scaler", RobustScaler()),
                     ("power", PowerTransformer(method="yeo-johnson"))])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(FEAT_PATH)
    df["clean_text"] = df["clean_text"].fillna("")
    feature_cols = ["clean_text"] + F.NUM_COLS
    X = df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df["Pattern Category"])
    print(f"Loaded {len(df)} rows, {len(le.classes_)} classes.")

    # ---- held-out test set carved out FIRST --------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_tr)}  Test (held-out): {len(X_te)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- model zoo (SMOTE inside each fold via imblearn pipeline) ----------
    models = {
        "Logistic Regression": ImbPipeline([
            ("prep", make_preprocessor(std_numeric())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE))]),
        "Linear SVC": ImbPipeline([
            ("prep", make_preprocessor(std_numeric())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))]),
        "Complement NB": ImbPipeline([
            ("prep", make_preprocessor(MinMaxScaler())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", ComplementNB())]),
        "Random Forest": ImbPipeline([
            ("prep", make_preprocessor(std_numeric())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        "XGBoost": ImbPipeline([
            ("prep", make_preprocessor(std_numeric())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE,
                                  tree_method="hist"))]),
    }

    print("\n=== 5-fold CV (macro-F1) on TRAIN split ===")
    cv_rows = []
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
        cv_rows.append({"Model": name, "CV Macro-F1": scores.mean(), "CV Std": scores.std()})
        print(f"  {name:22} macro-F1 = {scores.mean():.4f} +/- {scores.std():.4f}")
    cv_df = pd.DataFrame(cv_rows).sort_values("CV Macro-F1", ascending=False)
    cv_df.to_csv(os.path.join(REPORTS_DIR, "cv_model_comparison.csv"), index=False)

    # ---- Optuna tuning of XGBoost using CV macro-F1 on TRAIN only ----------
    print("\n=== Optuna tuning XGBoost (CV macro-F1, train only) ===")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
        )
        pipe = ImbPipeline([
            ("prep", make_preprocessor(std_numeric())),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(**params, eval_metric="mlogloss",
                                  random_state=RANDOM_STATE, tree_method="hist"))])
        return cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    print(f"  Best CV macro-F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # ---- fit tuned model on full TRAIN, evaluate ONCE on held-out TEST -----
    best = ImbPipeline([
        ("prep", make_preprocessor(std_numeric())),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(**study.best_params, eval_metric="mlogloss",
                              random_state=RANDOM_STATE, tree_method="hist"))])
    best.fit(X_tr, y_tr)
    pred = best.predict(X_te)
    test_acc = accuracy_score(y_te, pred)
    test_f1 = f1_score(y_te, pred, average="macro")
    print(f"\n=== HELD-OUT TEST (tuned XGBoost) ===")
    print(f"  Accuracy: {test_acc:.4f}   Macro-F1: {test_f1:.4f}")

    report = classification_report(y_te, pred, target_names=le.classes_, zero_division=0)
    print(report)
    with open(os.path.join(REPORTS_DIR, "test_classification_report.txt"), "w") as fh:
        fh.write(f"Held-out test accuracy: {test_acc:.4f}\n")
        fh.write(f"Held-out test macro-F1: {test_f1:.4f}\n\n")
        fh.write(report)

    # confusion matrix plot
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=90, fontsize=7)
    ax.set_yticklabels(le.classes_, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — tuned XGBoost (test macro-F1={test_f1:.3f})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ---- binary violation model (held-out eval) ----------------------------
    yb = df["label"].astype(int).values
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X, yb, test_size=0.20, stratify=yb, random_state=RANDOM_STATE)
    bin_pipe = ImbPipeline([
        ("prep", make_preprocessor(std_numeric())),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss",
                              random_state=RANDOM_STATE, tree_method="hist"))])
    bin_pipe.fit(Xb_tr, yb_tr)
    bpred = bin_pipe.predict(Xb_te)
    bin_report = classification_report(yb_te, bpred, zero_division=0)
    print("=== Binary violation model (held-out test) ===")
    print(bin_report)
    with open(os.path.join(REPORTS_DIR, "binary_classification_report.txt"), "w") as fh:
        fh.write(bin_report)

    # ---- refit on ALL data for deployment & save --------------------------
    best.fit(X, y)
    bin_pipe.fit(X, yb)
    joblib.dump(best, os.path.join(MODELS_DIR, "best_multi_model.joblib"))
    joblib.dump(bin_pipe, os.path.join(MODELS_DIR, "best_binary_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as fh:
        json.dump({
            "n_rows": int(len(df)),
            "n_classes": int(len(le.classes_)),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "best_params": study.best_params,
            "best_cv_macro_f1": float(study.best_value),
            "cv_comparison": cv_df.to_dict(orient="records"),
        }, fh, indent=2)
    print("\nSaved models to models/ and reports to reports/.")


if __name__ == "__main__":
    main()
