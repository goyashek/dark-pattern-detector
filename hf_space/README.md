---
title: Dark Pattern Detector (DistilBERT)
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.6.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Fine-tuned DistilBERT flagging deceptive UI copy (CCPA 2023)
---

# Dark Pattern Detector — DistilBERT

Fine-tuned DistilBERT that classifies UI copy into **India's CCPA Dark Pattern
Guidelines, 2023** taxonomy — 13 illegal dark-pattern classes plus a benign class.

This Space is the higher-accuracy companion to a classical TF-IDF + engineered-feature
model (served separately as a Streamlit app). The classical model is the fast, fully
interpretable default; this transformer reads whole-phrase meaning and holds up better on
real-world text, at the cost of size and explainability.

## Why no keyword badges?

The classical app highlights *which* hand-built signals fired (urgency words, hidden-fee
phrasing, cancellation friction). DistilBERT has no such features — so its honest
interpretability surface is the **full confidence distribution across all 14 classes**,
shown alongside each prediction. When the top dark-pattern class is below the confidence
threshold, it defaults to *benign* to favour precision over false alarms.

## Configuration

- Runs on **CPU Basic** (free). Single-example inference is interactive (~tens of ms);
  a GPU offers no perceptible benefit at this scale.
- Weights load from a separate model repo via `MODEL_ID` (default
  `goyashek/distilbert-darkpattern`). Override with a Space variable if your repo differs.

## Disclaimer

Research / educational demo — not legal advice. Made by
[Abhishek Goyal](https://github.com/goyashek).
