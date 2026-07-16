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
short_description: DistilBERT screening for possible dark-pattern text
---

# Dark Pattern Detector: DistilBERT

This Space runs the DistilBERT model I fine-tuned for my dark-pattern text project. It predicts one of the 13 categories named in India's 2023 CCPA dark-pattern guidelines or the no-dark-pattern class.

The model only reads UI text. It cannot inspect visual hierarchy, default selections, repeated prompts, or a complete user flow.

## Model scores

The classical version of this project has 12 explicit text features. DistilBERT does not use those features, so this app shows its top softmax scores instead. The scores are not calibrated confidence.

A top score below the provisional 50% display threshold is reported as inconclusive. It is not changed to a benign prediction.

## Configuration

The Space runs on CPU Basic. It loads weights from `goyashek/distilbert-darkpattern` by default. The `MODEL_ID` Space variable can point to a different repository or local folder.

## Disclaimer

This is a student project. The category mapping reflects my own reading of the CCPA dark-pattern guidelines. It is not official or approved by the CCPA, and the results should not be used as legal or compliance advice.

[Abhishek Goyal](https://github.com/goyashek)
