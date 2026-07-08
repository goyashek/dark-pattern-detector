"""error_analysis.py — Phase 4 diagnostics (read-only, not committed).

Loads the leak-free split, retrains the two classical models on it (fast, same NB2/NB3
pipeline), loads the fine-tuned DistilBERT artifact, and reports WHERE each model breaks:
per-class F1 in-distribution, the top confusion pairs, and a row-by-row OOD table (23 rows,
small enough to eyeball). Goal: decide the improvement track from evidence, not guesswork.

Run:  /opt/anaconda3/bin/python -m src.error_analysis
"""
import json
import os
import re
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

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

SEED = 42
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_COLS = [
    "urgency_kw_count", "scarcity_kw_count", "shame_phrase_flag", "cancel_diff_score",
    "social_proof_flag", "price_drip_flag", "discount_claim_flag", "neg_option_flag",
    "all_caps_ratio", "exclamation_count", "question_count", "text_length", "word_count",
    "number_present", "time_reference_flag", "noun_ratio", "verb_ratio", "adj_ratio",
    "adv_ratio", "sentiment_polarity", "sentiment_subjectivity", "avg_word_len",
]

# ---- NB1 feature code (verbatim) so classical sees identical OOD inputs ----
URGENCY_KW = [r"hurry", r"limited time", r"ends in", r"only.*hours", r"today only", r"flash sale",
              r"act now", r"last chance", r"don't miss", r"expires", r"sale ends", r"ending soon"]
SCARCITY_KW = [r"only \d+ left", r"low stock", r"selling fast", r"almost gone", r"high demand",
               r"\d+ remaining", r"few left", r"running out", r"limited stock"]
SHAME_PHRASES = [r"no thanks", r"i don't need", r"prefer to pay", r"i hate", r"hate saving", r"i prefer"]
CANCEL_DIFF = [r"to cancel.*call", r"registered mail", r"hotline", r"cancellation fee",
               r"live agent", r"business hours", r"retention team", r"call.*to cancel"]
SOCIAL_PROOF = [r"people.*viewing", r"people.*bought", r"watching this", r"\d+ people", r"booked in the last"]
PRICE_DRIP = [r"processing fee", r"service charge", r"convenience fee", r"booking fee", r"resort fee",
              r"handling fee", r"surcharge", r"added at checkout", r"not included in"]
DISCOUNT = [r"\d+% off", r"save \$", r"you save", r"coupon", r"discount", r"deal", r"offer"]
NEG_OPT = [r"pre-ticked", r"auto-renew", r"pre-selected", r"checked by default", r"uncheck", r"automatically renew"]

_lemm = WordNetLemmatizer()
def clean_and_lemmatize(text):
    text = re.sub(r'[^a-zA-Z\s!?]', '', str(text).lower())
    return ' '.join(_lemm.lemmatize(t) for t in word_tokenize(text))

def extract_features(text):
    text = str(text); low = text.lower(); blob = TextBlob(text)
    tokens = word_tokenize(low); pos = nltk.pos_tag(tokens) if tokens else []
    total = len(pos) if pos else 1
    nouns = sum(1 for _, t in pos if t.startswith('NN')); verbs = sum(1 for _, t in pos if t.startswith('VB'))
    adjs = sum(1 for _, t in pos if t.startswith('JJ')); advs = sum(1 for _, t in pos if t.startswith('RB'))
    words = text.split()
    return {
        "urgency_kw_count": sum(bool(re.search(p, low)) for p in URGENCY_KW),
        "scarcity_kw_count": sum(bool(re.search(p, low)) for p in SCARCITY_KW),
        "shame_phrase_flag": int(any(re.search(p, low) for p in SHAME_PHRASES)),
        "cancel_diff_score": sum(bool(re.search(p, low)) for p in CANCEL_DIFF),
        "social_proof_flag": int(any(re.search(p, low) for p in SOCIAL_PROOF)),
        "price_drip_flag": int(any(re.search(p, low) for p in PRICE_DRIP)),
        "discount_claim_flag": int(any(re.search(p, low) for p in DISCOUNT)),
        "neg_option_flag": int(any(re.search(p, low) for p in NEG_OPT)),
        "all_caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "exclamation_count": text.count("!"), "question_count": text.count("?"),
        "text_length": len(text), "word_count": len(words),
        "number_present": int(bool(re.search(r"\d+", text))),
        "time_reference_flag": int(bool(re.search(r"hour|minute|day|today|soon|week|month|year", low))),
        "noun_ratio": nouns/total, "verb_ratio": verbs/total, "adj_ratio": adjs/total, "adv_ratio": advs/total,
        "sentiment_polarity": blob.sentiment.polarity, "sentiment_subjectivity": blob.sentiment.subjectivity,
        "avg_word_len": sum(len(w) for w in words)/len(words) if words else 0,
    }


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
