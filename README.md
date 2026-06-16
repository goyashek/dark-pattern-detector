# 🔍 CCPA 2023 Compliance Classifier: Dark Pattern Detector

<p align="left">
  <a href="https://dark-patterns.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/goyashek" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile">
  </a>
</p>

A compliance auditing tool that reads website/application UI text copy (e.g., urgency flags, pre-checked opt-ins, confirm-shaming prompts) and classifies it into one of **14 categories**—India's **13 illegal dark-pattern classes** established by the **Central Consumer Protection Authority (CCPA) in November 2023** plus a *Not a Dark Pattern* (safe/benign) class.

> [!NOTE]
> **Live Auditing Tool**: Interactive Streamlit dashboard is deployed at [dark-patterns.streamlit.app](https://dark-patterns.streamlit.app/)

---

## 👨‍💻 Author & Credits
- **Created By**: [Abhishek Goyal](https://github.com/goyashek)
- **GitHub**: [github.com/goyashek](https://github.com/goyashek)
- **Kaggle Source**: Indian context compliance dataset sourced from Kaggle: **[Insert Kaggle Dataset Link Here]**

---

## 🇮🇳 Regulatory Context & Motivation

> [!IMPORTANT]
> On **30 November 2023**, India's CCPA declared **13 categories of dark patterns** illegal under the Consumer Protection Act, 2019. 
> This tool automates the compliance review of UI text strings against these enforceable legal clauses.

---

## 🔬 Bridging Academic Taxonomy and Legal Reality

This project maps the academic baseline corpus (**Yada et al. 2022**) onto CCPA legal compliance classes.

### Comparative Analysis: Baseline vs. This Project

| Feature / Metric | Yada et al. 2022 (Baseline) | This Project |
| :--- | :--- | :--- |
| **Label Space** | Binary + 7 Academic Taxonomy Classes | **14 Classes** (13 CCPA Legal Classes + Benign) |
| **Practical Context** | Academic Research | **Regulatory Compliance & Auditing** |
| **Class Coverage** | Missing legal categories, high class skew | **All 13 CCPA classes represented** |
| **Explainability** | Black-box Transformer predictions | **Interpretable NLP features** + active lexical badge triggers |
| **Inference Layer** | Raw uncalibrated model outputs | **Precision-gate thresholding + Toned-down UI confidence** |

---

## 🗺️ Pipeline & Flowchart

```mermaid
flowchart TD
    A[Yada-2022 Raw Corpus] --> C[Remap Academic -> 13 CCPA classes]
    B[Kaggle Indian Compliance Data] --> D[Merge Datasets]
    C --> D
    D --> E[Global De-duplication Leakage Fix]
    E --> F[22 Interpretable NLP Features + TF-IDF]
    F --> G[ColumnTransformer Pipeline]
    G --> H[SMOTE fit inside CV Folds]
    H --> I[Model Zoo Evaluation]
    I --> J[Optuna-Tuned XGBoost]
    J --> K[Precision Gate & Calibration Layer]
    K --> L[Export Joblib -> Streamlit App]
```

---

## 🛠️ Advanced Techniques & Rationale (Why We Did Them)

### 1. Global De-duplication Before Splitting
* **Why**: Prevents identical UI text strings from appearing in both train and test sets, avoiding data leakage and artificially inflated accuracy.

### 2. SMOTE Oversampling Inside Cross-Validation Folds
* **Why**: Solves class imbalance for rare classes (like *Subscription Trap*) without leaking validation partition data into the training process.

### 3. Robust Scaling and Power Transformation (Yeo-Johnson)
* **Why**: Normalizes feature distributions and dampens outliers for highly skewed metrics (like capitalization ratios and text lengths).

### 4. Cross-Validated Macro-F1 Optimization via Optuna
* **Why**: Forces the hyperparameter search to weight minority classes equally, preventing class neglect that occurs when optimizing for simple accuracy.

### 5. Shared Feature Extraction Module
* **Why**: Eliminates train-serve skew by ensuring text processing is performed identically in both training and production.

### 6. Inference-Time Precision Gate
* **Why**: Minimizes false positives by defaulting predictions with low classifier confidence (< 65%) to "Not a Dark Pattern".

### 7. UI Confidence Calibration
* **Why**: Tones down overconfident probability scores (99%+) in the Streamlit interface to display realistic compliance scores.

---

## 📂 Codebase Architecture

```
dark-pattern-pro/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_nlp_eda.ipynb         # EDA, tokenization & keyword extraction
│   └── 02_model_tuning_export.ipynb  # Cross-validation, Optuna tuning & export
├── src/
│   ├── features.py         # Shared module: cleans text, POS tagger & extracts 22 features
│   ├── collect_data.py     # Aggregates manually gathered legal compliance examples
│   ├── build_dataset.py    # Merges data sources, remaps academic -> legal labels & de-duplicates
│   ├── make_features.py    # Transforms text files into tabular dataset (features.csv)
│   └── train.py            # Automated script mirror of the notebook modeling pipeline
├── data/
│   ├── raw/dataset_raw.tsv         # Original Yada et al. e-commerce dataset
│   └── processed/ccpa_dataset.tsv  # Final cleaned & remapped corpus
├── models/
│   ├── best_multi_model.joblib     # Tuned 14-class XGBoost model
│   ├── best_binary_model.joblib    # 2-class violation/benign model
│   └── label_encoder.joblib        # Target Class encoder
└── app/
    └── app.py              # Streamlit compliance dashboard
```

---

## 🚀 How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Reproduce the Modeling Pipeline
Run Jupyter Notebooks:
```bash
jupyter notebook notebooks/01_data_nlp_eda.ipynb
jupyter notebook notebooks/02_model_tuning_export.ipynb
```
Or run python scripts:
```bash
python -m src.collect_data
python -m src.build_dataset
python -m src.make_features
python -m src.train
```

### Launch the Dashboard Local Server
```bash
streamlit run app/app.py
```

---

## 🔬 The 22 Engineered NLP Features
- **Lexical/Keyword Triggers**: urge_kw_count, scarcity_kw_count, shame_phrase_flag, cancel_diff_score, social_proof_flag, price_drip_flag, discount_claim_flag, neg_option_flag.
- **Structural Indicators**: all_caps_ratio, exclamation_count, question_count, text_length, word_count, number_present, time_reference_flag.
- **Part-of-Speech (POS) Mix**: noun_ratio, verb_ratio, adj_ratio, adv_ratio.
- **TextBlob Sentiment**: sentiment_polarity, sentiment_subjectivity, and average_word_length.

---

## ⚠️ Disclaimer
> [!WARNING]
> This is a **test demonstration project** and does not constitute formal legal advice. Always consult a legal professional for compliance validation.
