"""error_analysis.py — Phase 4 diagnostics (read-only, not committed).

Loads the leak-free split, retrains the two classical models on it (fast, same NB2/NB3
pipeline), loads the fine-tuned DistilBERT artifact, and reports WHERE each model breaks:
per-class F1 in-distribution, the top confusion pairs, and a row-by-row OOD table (23 rows,
small enough to eyeball). Goal: decide the improvement track from evidence, not guesswork.

Run:  /opt/anaconda3/bin/python -m src.error_analysis
"""
import json
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src import features as F

SEED = 42
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_COLS = F.NUM_COLS

def clean_and_lemmatize(text):
    return F.clean_and_lemmatize(text)

def extract_features(text):
    return F.extract_features(text)


def classical_pipe(kind):
    prep = ColumnTransformer([
        ('text', TfidfVectorizer(max_features=300, ngram_range=(1, 3), min_df=2, sublinear_tf=True), 'clean_text'),
        ('num', Pipeline([('scale', RobustScaler()), ('power', PowerTransformer(method='yeo-johnson'))]), NUM_COLS),
    ])
    if kind == 'svc':
        clf = LinearSVC(C=2.1243, loss='squared_hinge', class_weight='balanced', random_state=SEED, max_iter=5000)
    else:
        clf = XGBClassifier(n_estimators=105, max_depth=9, learning_rate=0.20397, subsample=0.7637,
                            colsample_bytree=0.7545, eval_metric='mlogloss', random_state=SEED, tree_method='hist')
    return ImbPipeline([('prep', prep), ('smote', SMOTE(random_state=SEED)), ('clf', clf)])


def bert_predict(texts, model_dir, classes):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device).eval()
    enc = tok(list(texts), truncation=True, padding='max_length', max_length=64, return_tensors='pt')
    with torch.no_grad():
        logits = model(input_ids=enc['input_ids'].to(device),
                       attention_mask=enc['attention_mask'].to(device)).logits
    return logits.argmax(1).cpu().numpy()


def main():
    df = pd.read_csv(os.path.join(HERE, 'data/processed/features.csv'))
    df['clean_text'] = df['clean_text'].fillna('')
    split = json.load(open(os.path.join(HERE, 'reports/leak_free_split.json')))
    tr, te = np.array(split['train']), np.array(split['test'])

    le = LabelEncoder()
    y = le.fit_transform(df['Pattern Category'])
    CLASSES = list(le.classes_)
    X = df[['clean_text'] + NUM_COLS]

    # train classical on leak-free split
    pipes = {}
    preds = {}
    for kind, name in [('svc', 'SVC'), ('xgb', 'XGB')]:
        p = classical_pipe(kind); p.fit(X.iloc[tr], y[tr]); pipes[name] = p
        preds[name] = p.predict(X.iloc[te])

    # DistilBERT on the same test rows
    bert_dir = os.path.join(HERE, 'models/distilbert_darkpattern')
    preds['BERT'] = bert_predict(df['text'].iloc[te].tolist(), bert_dir, CLASSES)
    y_te = y[te]

    print("=" * 70)
    print("IN-DISTRIBUTION (leak-free test, n=%d) — per-class F1" % len(te))
    print("=" * 70)
    rows = []
    support = pd.Series(y_te).value_counts().to_dict()
    for ci, cls in enumerate(CLASSES):
        row = {'class': cls, 'n': support.get(ci, 0)}
        for m in ['SVC', 'XGB', 'BERT']:
            mask = (y_te == ci)
            row[m] = round(f1_score((y_te == ci), (preds[m] == ci), zero_division=0), 3)
        rows.append(row)
    perclass = pd.DataFrame(rows).sort_values('BERT')
    print(perclass.to_string(index=False))
    print("\nmacro-F1:", {m: round(f1_score(y_te, preds[m], average='macro', zero_division=0), 3)
                          for m in ['SVC', 'XGB', 'BERT']})

    # top confusion pairs for BERT (the headline model)
    print("\n" + "=" * 70)
    print("TOP CONFUSIONS — DistilBERT (true -> predicted, off-diagonal)")
    print("=" * 70)
    cm = confusion_matrix(y_te, preds['BERT'])
    conf = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                conf.append((cm[i, j], CLASSES[i], CLASSES[j]))
    for n, a, b in sorted(conf, reverse=True)[:12]:
        print(f"  {n:3d}  {a:28s} -> {b}")

    # ---- OOD row-by-row ----
    ood = pd.read_csv(os.path.join(HERE, 'data/processed/ood_real_test.csv'))
    ood = ood[ood['Pattern Category'].isin(CLASSES)].reset_index(drop=True)
    ood['clean_text'] = ood['text'].apply(clean_and_lemmatize)
    ood_feats = pd.DataFrame([extract_features(t) for t in ood['text']])
    ood_X = pd.concat([ood['clean_text'], ood_feats], axis=1)
    y_ood = le.transform(ood['Pattern Category'])

    ood_pred = {m: pipes[m].predict(ood_X) for m in ['SVC', 'XGB']}
    ood_pred['BERT'] = bert_predict(ood['text'].tolist(), bert_dir, CLASSES)

    print("\n" + "=" * 70)
    print("OOD (23 real rows) — row-by-row  [✓ = correct]")
    print("=" * 70)
    for i in range(len(ood)):
        true = CLASSES[y_ood[i]]
        marks = {m: ('✓' if ood_pred[m][i] == y_ood[i] else CLASSES[ood_pred[m][i]][:10]) for m in ['SVC','XGB','BERT']}
        print(f"[{ood['text'][i][:46]:46s}] T={true[:14]:14s} "
              f"S:{marks['SVC']:11s} X:{marks['XGB']:11s} B:{marks['BERT']:11s}")
    print("\nOOD macro-F1:", {m: round(f1_score(y_ood, ood_pred[m], average='macro', zero_division=0), 3)
                              for m in ['SVC', 'XGB', 'BERT']})

    # OOD vocabulary overlap diagnostic (classical's weakness)
    train_vocab = set()
    for t in df['clean_text'].iloc[tr]:
        train_vocab.update(t.split())
    oov_rates = []
    for t in ood['clean_text']:
        toks = t.split()
        if toks:
            oov_rates.append(sum(1 for w in toks if w not in train_vocab) / len(toks))
    print(f"\nOOD out-of-train-vocab rate (clean_text tokens): mean={np.mean(oov_rates):.2f}")
    print(f"OOD avg word_count={ood['text'].str.split().apply(len).mean():.1f} "
          f"vs train avg={df['text'].iloc[tr].str.split().apply(len).mean():.1f}")


if __name__ == "__main__":
    main()
