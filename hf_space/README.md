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
short_description: DistilBERT risk screening for potential dark-pattern text
---

# Dark Pattern Detector — DistilBERT

Fine-tuned DistilBERT research screener that maps UI copy onto the 13 categories named in
**India's CCPA Dark Pattern Guidelines, 2023**, plus a no-dark-pattern class.

This Space is the higher-accuracy companion to a classical TF-IDF + engineered-feature
model (served separately as a Streamlit app). The classical model is the fast, fully
interpretable default; this transformer reads whole-phrase meaning and holds up better on
real-world text, at the cost of size and explainability.

## Why no keyword badges?

The classical app highlights *which* hand-built signals fired (urgency words, hidden-fee
phrasing, cancellation friction). DistilBERT has no such features — so its honest
interpretability surface is the **full softmax score distribution across all 14 classes**,
shown alongside each prediction. These scores are not calibrated confidence. Any top score
below the provisional 50% display threshold is reported as *inconclusive* rather than being
converted to benign.

## Configuration

- Runs on **CPU Basic** (free). Single-example inference is interactive (~tens of ms);
  a GPU offers no perceptible benefit at this scale.
- Weights load from a separate model repo via `MODEL_ID` (default
  `goyashek/distilbert-darkpattern`). Override with a Space variable if your repo differs.

## Disclaimer

This is a student project. I mapped the dataset labels to the CCPA dark-pattern categories
based on my own reading of the guidelines. The mapping is not official or approved by the
CCPA, and the results should not be used as legal or compliance advice. Made by
[Abhishek Goyal](https://github.com/goyashek).
