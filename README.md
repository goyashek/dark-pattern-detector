# 🔍 CCPA 2023 Compliance Classifier: Dark Pattern Detector

[![Streamlit App](https://static.streamlit.io/badge-gradient.svg)](https://dark-patterns.streamlit.app/)
[![GitHub Profile](https://img.shields.io/badge/GitHub-goyashek-orange?style=flat&logo=github)](https://github.com/goyashek)
[![CCPA-2023 Enforceable](https://img.shields.io/badge/CCPA--2023-Enforceable-red?style=flat)](https://github.com/goyashek)

**Route 1: Classical NLP + Core ML Pipeline & Auditing Dashboard**

A compliance auditing tool that reads website/application UI text copy (e.g., urgency flags, pre-checked opt-ins, confirm-shaming prompts) and classifies it into one of **14 categories**—India's **13 illegal dark-pattern classes** established by the **Central Consumer Protection Authority (CCPA) in November 2023** plus a *Not a Dark Pattern* (safe/benign) class.

---

## 🚀 Live Demo & Deployment
You can interact with the live auditing tool here:  
👉 **[https://dark-patterns.streamlit.app/](https://dark-patterns.streamlit.app/)**

---

## 👨‍💻 Author & Credits
- **Created By**: [Abhishek Goyal](https://github.com/goyashek)
- **GitHub**: [github.com/goyashek](https://github.com/goyashek)

---

## 🇮🇳 Regulatory Context & Motivation

On **30 November 2023**, India's CCPA notified new guidelines declaring **13 categories of dark patterns** illegal under the Consumer Protection Act, 2019. E-commerce platforms, booking engines, and SaaS providers operating in India face regulatory penalties for violating these guidelines. 

Given the millions of UI copy changes pushed daily across the web, manual compliance auditing is impossible. This project bridges the gap between **legal requirements** and **automated machine learning auditing** by building a classifier that maps arbitrary UI text onto enforceable legal clauses.

---

## 🔬 Bridging Academic Taxonomy and Legal Reality

This project takes the raw dataset from the academic baseline **Yada et al. 2022** (which classifies dark patterns into Mathur's academic taxonomy) and maps it onto the 2023 CCPA legal clauses.

### Comparative Analysis: Baseline vs. This Project

| Metric/Feature | Yada et al. 2022 (Baseline) | This Project |
| :--- | :--- | :--- |
| **Label Space** | Binary + 7 Academic Taxonomy Classes | **14 Classes** (13 CCPA Legal Classes + Benign) |
| **Practical Context** | Academic Research | **Regulatory Compliance & Auditing** |
| **Class Coverage** | Missing legal categories, high class skew | **All 13 CCPA classes represented** using custom examples |
| **Explainability** | Black-box Transformer predictions | **Interpretable NLP features** + active lexical badge triggers |
| **Inference Layer** | Raw uncalibrated model outputs | **Precision-gate thresholding + Toned-down UI confidence** |

---

## 🛠️ Advanced Techniques & Rationale (Why We Did Them)

To build a robust classifier suitable for real-world auditing, we implemented several advanced machine learning techniques. Below is the rationale for each engineering decision:

### 1. Global De-duplication Prior to Splitting
* **Why**: To prevent data leakage. UI copy contains frequent duplicate strings (e.g., standard cancel buttons or cookie consent alerts). Splitting the dataset before de-duplicating would cause identical text to populate both the training and test folds. This would inflate validation accuracy, leading to a model that has memorized specific strings rather than generalizing to new ones.

### 2. SMOTE Oversampling Inside Cross-Validation Folds
* **Why**: To fix severe class imbalance without validation bias. Rare classes like *Subscription Trap* have very few samples. Using SMOTE on the entire dataset prior to splitting leaks synthetic samples generated from validation data into the training fold. Wrapping the preprocessing in an `imblearn.Pipeline` ensures that SMOTE is executed strictly on the training partition of each fold during cross-validation.

### 3. Robust Scaling and Power Transformation (Yeo-Johnson)
* **Why**: To stabilize variance and normalize feature distributions. Engineered structural NLP features (such as text length and punctuation counts) follow highly skewed, long-tailed distributions. Applying a `RobustScaler` reduces the influence of extreme outlier copy, while the Yeo-Johnson transformer shifts numeric inputs closer to a normal distribution, improving the convergence of linear baselines.

### 4. Cross-Validated Macro-F1 Optimization via Optuna
* **Why**: To prevent minority class neglect. In highly imbalanced datasets, overall accuracy can be high even if rare categories are misclassified. Tuning our XGBoost hyperparameters using Optuna on Macro-F1 forces the optimization search to weight minority classes equally, ensuring the model learns the nuances of rare dark patterns.

### 5. Shared Feature Extraction Module
* **Why**: To eliminate train-serve skew. In ML engineering, redefining text cleaning or tokenization logic for production often introduces subtle bugs. By having both the training pipeline and the Streamlit app import from `src/features.py`, we guarantee that inference features are extracted identically to how they were trained.

### 6. Inference-Time Precision Gate
* **Why**: To minimize false positives in auditing. Standard classifiers flag violations using simple argmax, which defaults to the class with the highest score even if confidence is low (e.g., 36%). In a compliance tool, false positives disrupt the auditor's workflow. We enforce a $0.65$ probability threshold for dark patterns; predictions failing to meet this gate revert to `"Not a Dark Pattern"`.

### 7. UI Confidence Calibration
* **Why**: To present realistic uncertainty. Raw classifiers often output overconfident predictions ($99.9\%+$ confidence). We adjust the confidence outputs shown in the Streamlit UI to tone down overconfidence, aligning predictions with real-world user expectations and making the compliance dashboard less alarmist.

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
│   └── label_encoder.joblib        # Scikit-learn Target Class Encoder
└── app/
    └── app.py              # Streamlit compliance dashboard
```

---

## 🚀 How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Reproduce the Notebook Pipeline
You can run the Jupyter notebooks in order to reproduce the exploratory data analysis, feature engineering, and model training:
```bash
jupyter notebook notebooks/01_data_nlp_eda.ipynb
jupyter notebook notebooks/02_model_tuning_export.ipynb
```

Alternatively, you can run the pipeline directly via terminal scripts:
```bash
python -m src.collect_data
python -m src.build_dataset
python -m src.make_features
python -m src.train
```

### Launch the Auditing Dashboard
Run the Streamlit server locally to launch the UI dashboard:
```bash
streamlit run app/app.py
```

---

## 🔬 The 22 Engineered NLP Features
The model operates on a hybrid features space consisting of **TF-IDF n-grams (max 300 features)** and **22 hand-engineered features** representing:
- **Lexical/Keyword Triggers**: urge_kw_count, scarcity_kw_count, shame_phrase_flag, cancel_diff_score, social_proof_flag, price_drip_flag, discount_claim_flag, neg_option_flag.
- **Structural Indicators**: all_caps_ratio, exclamation_count, question_count, text_length, word_count, number_present, time_reference_flag.
- **Part-of-Speech (POS) Mix**: noun_ratio, verb_ratio, adj_ratio, adv_ratio.
- **TextBlob Sentiment**: sentiment_polarity, sentiment_subjectivity, and average_word_length.

---

## ⚠️ Limitations & Real-World Generalization
- **Class Skew in Rare Categories**: Some CCPA categories (e.g., *Disguised Advertisement*, *Rogue Malware*) are rarely found in public e-commerce datasets. While manually-collected samples and SMOTE stabilize predictions, these classes have smaller support.
- **Out-of-Distribution Inputs**: Brand new deceptive phrasing that falls far outside the current training distribution can bypass keywords and TF-IDF ranges. 
- **Not Legal Advice**: This software is designed for auditing assistance. Final compliance decisions should always be reviewed by a human legal counsel.
