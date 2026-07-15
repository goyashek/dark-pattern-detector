# Dark Pattern Detector: Complete Project Audit and Plan of Action

**Audit date:** 2026-07-15
**Repository state:** `main` at `053a1b1`
**Scope:** product claims, regulatory framing, data, labeling, leakage, evaluation,
modeling, inference, applications, dependencies, reproducibility, testing, deployment,
security, accessibility, documentation, and repository hygiene.

## Session 5 Completion Update — 2026-07-15

This section is the handoff point for the next session. It supersedes older current-state
statements below; the original findings remain as the audit trail.

### Completed in this session

- Replaced the stale random split with a content-hashed 5,051/1,322 split over connected
  `page_id` or template-skeleton groups. Both outer partitions and every inner fold have zero
  group overlap and all 14 classes.
- Selected the requested classical variant: character 2-6 gram TF-IDF, 12 engineered features,
  RobustScaler + Yeo-Johnson, SMOTE, LinearSVC (`C=0.5`), and three-fold grouped sigmoid
  calibration.
- Exported the calibrated classical artifact and label encoder with scikit-learn 1.7.2. Its
  stored evaluation is `0.739 ± 0.059` grouped training CV, `0.732` test macro-F1, `0.825`
  accuracy, and `0.517` OOD-development macro-F1.
- Removed the Streamlit fallback threshold and probability damping. The app now uses the exact
  12-feature contract and shows the calibrated winning-class probability directly.
- Reduced the active feature table from the stale 22-feature block to exactly 12 engineered
  columns. `features.csv` now has 6,373 rows and 17 total columns; `ood_features.csv` has 23 rows
  and the same schema.
- Made all three notebooks self-contained. They no longer import project scripts or shell out to
  training commands. Notebook 2 and Notebook 3 contain an identical classical grouping/model
  block, and the duplicate Notebook 2 under `src/` is byte-identical.
- Reran all three notebooks. The latest same-split Notebook 3 comparison is:

| Model | Test macro-F1 | Test accuracy | OOD-dev macro-F1 |
|---|---:|---:|---:|
| DistilBERT | 0.883 | 0.911 | 0.697 |
| Classical 12-feature model | 0.730 | 0.816 | 0.517 |
| LSTM from scratch | 0.657 | 0.784 | 0.389 |

- Added small regression tests for the exact 12-feature contract and transitive page/template
  grouping. Notebook format, AST, execution counts, saved outputs, model inference, Streamlit
  startup, data hash, split invariants, and Git whitespace checks passed during the session.

### Audit items materially improved

- **P0-3 / A06:** current classical inner and outer validation are group-aware.
- **P0-5 / A04:** the Streamlit artifact, feature contract, README, and stored classical metrics
  now describe the calibrated character-SVC pipeline rather than the old XGBoost artifact.
- **P0-6:** the classical requirement and exported artifact now agree on scikit-learn 1.7.2.
- **P0-7 / A02:** fake Streamlit confidence transformation was removed. The HF Space threshold
  path still needs review.
- **P1-1 / A13:** the saved split now includes a dataset SHA-256 and validates coverage,
  disjointness, ordering, and page/template overlap.
- **P1-3:** the notebook flow is now self-contained and consistent. Script/notebook duplication
  remains deliberate because the notebooks were explicitly requested to run independently.
- **P1-5:** a minimal test suite exists, but CI and broader data/inference coverage remain open.

### Start here next session

1. Complete **A01/A02**: replace legal verdict wording with risk-screening language, add
   insufficient-context guidance, and remove or justify the HF Space confidence fallback.
2. Complete **A03/A05**: add an artifact manifest and exact tested environment/revision metadata,
   including the newly rerun DistilBERT artifact.
3. Address **A10-A12**: validate taxonomy mapping and provenance, then plan a real adjudicated
   benchmark that excludes generated rows from validation/test.
4. Expand **A09**: add CI, artifact-load/golden-prediction checks, and app smoke tests.

Do not tune further against the 23-row OOD development set. The next useful gains are product
honesty, artifact traceability, and real evaluation data.

## 0. July 15 Verification Update

This update records the full file-by-file pass. It supersedes older statements in this
document where they conflict.

- The working tree contained 208 project files excluding `.git`: 42 tracked and 166 ignored or
  generated. Python, JSON, YAML, notebooks, images, and serialized artifacts were structurally
  checked; no secrets or corrupt files were found.
- The supposed second/Kaggle dataset is not a distinct external dataset in this checkout.
  `src.collect_data.build_collected()` reproduces all 4,217 rows in
  `data/processed/collected.tsv`, and `data/raw/pattern_label.csv` contains the same generated
  rows apart from its index column. Documentation must describe Kaggle as the acquisition route
  only if there is evidence for it; the checked-in rows are generated by this repository.
- The OOD file has 23 rows across only 8 of 14 classes and no benign cases. All 23 final labels
  equal the scraper's automatic label, while only 12 candidate rows have a non-empty human
  `verified_label`. One OOD string, `Only 3 left`, also appears in the training corpus after case
  normalization. The OOD set is development data, not an independent test.
- The current outer split is skeleton-clean, but 227 of 412 real-origin test rows share a
  `page_id` with training across 111 pages. Future grouping must keep connected components formed
  by either template skeleton or source/page ID together.
- The official taxonomy mapping still maps every Social Proof row to Disguised Advertisement;
  309 retained rows are affected. This requires human review rather than another model tweak.
- The current deployed XGBoost artifact reproduces the stored 0.503 grouped-test macro-F1. The
  README's XGBoost 0.562 and SVC 0.576 belong to different notebook pipelines/artifacts.
- `models/*.joblib` embeds scikit-learn 1.7.2, `src/*.joblib` embeds 1.8.0, and the old local
  baseline embeds 1.6.1. The root requirement currently pins 1.6.1.
- Tracked Notebook 2 has mostly blank execution counts and stale random-split results. Its copy
  under `src/` also has undefined names. Notebook 1 and Notebook 3 are structurally much cleaner.
- The Streamlit threshold path lowers grouped OOD macro-F1 for the deployed XGBoost from about
  0.400 raw to 0.280 and can report non-benign probability mass as benign confidence.
- The workspace is roughly 807 MB mainly because the transformer and old artifacts are duplicated;
  the Git object store itself is only about 11 MB. Cleanup is useful but comes after evaluation
  integrity.

### Minimum execution order

1. Correct provenance, legal wording, OOD status, and deployed metric claims.
2. Replace all inner and outer grouping with stable connected groups over template and page/source;
   freeze the current test and stop inspecting it during model selection.
3. Try the character-TF-IDF linear baseline below on training-only grouped CV. Fix labels and add
   real adjudicated development data before doing broader tuning.
4. Export one artifact from one script, remove the fake confidence transformation, add a small
   data/split/inference check, and only then refresh README results.

Everything else in the longer backlog is optional until these four steps are complete.

## 1. Executive Verdict

This is a credible **research and portfolio prototype** with a useful core story: a
template-aware holdout exposes how badly naive random splitting inflated performance, and a
fine-tuned transformer performs better than classical text models on the project's current
benchmarks. The repository also contains several good engineering choices, including shared
classical feature extraction, a human-review boundary in the scraper, and separate lightweight
and transformer deployment surfaces.

It is **not yet a defensible compliance auditing tool**. The primary blockers are not model
architecture. They are ground-truth validity, benchmark independence, deployment/reporting
drift, legal overclaiming, and the absence of a reproducible tested release process.

The most important conclusions are:

1. **The legal claim exceeds the input and evidence.** Several CCPA dark patterns are
   properties of a flow, default state, visual hierarchy, repeated prompt, or price change.
   A single text string cannot establish them. The apps nevertheless announce a "CCPA
   violation" and present Annexure categories as numbered legal clauses.
2. **The training data is mostly generated.** 4,217 of 6,373 rows (66.17%) come from
   `src/collect_data.py`, despite descriptions such as "manually-collected". Four classes are
   100% generated and several others are approximately 98-99% generated.
3. **The outer split is template-clean, but model selection is not.** Ordinary stratified
   cross-validation is used inside the grouped training partition. This lets template siblings
   cross CV folds and produces CV macro-F1 near 0.96 while the grouped holdout score stored for
   the deployed XGBoost model is 0.503.
4. **The 23-row OOD benchmark is a development set, not a final test set.** It covers only 8
   of 14 labels, has no benign rows, and its observed language was used to modify synthetic
   training generators. It has also informed model regularization decisions.
5. **The deployed classical artifact is not the model described by the headline table.** The
   Streamlit app loads a 354-tree XGBoost pipeline evaluated at 0.503 macro-F1. The README
   reports 0.562 for another XGBoost experiment and highlights a 0.576 SVC that is not deployed.
6. **Artifact and runtime versions contradict each other.** The deployed joblib contains
   scikit-learn 1.7.2 objects while `requirements.txt` pins 1.6.1. The stale copies under `src/`
   contain 1.8.0 objects. The transformer config records
   Transformers 5.12.1 while the Space installs `transformers>=4.40,<5.0`.
7. **Confidence is not calibrated.** Both apps apply arbitrary thresholds; the Streamlit app
   additionally transforms probabilities and can display a fabricated benign confidence.
8. **There is no automated test suite or CI quality gate.** The only GitHub workflow deploys
   the HF Space directly after relevant pushes.

### Recommended product position

Reframe the project as a **dark-pattern risk screener and research demonstrator**. It should
identify potentially suspicious language and request the additional context required for a
human review. It should not declare a legal violation from one text snippet.

### Recommended technical direction

Do not start by adding a larger model. First build a provenance-rich, human-annotated, locked
evaluation set and a group-aware validation pipeline. Then calibrate and compare a strong
linear baseline with the transformer. Add contextual UI/flow inputs for categories that text
alone cannot determine.

## 2. Audit Method and Limitations

### Evidence reviewed

- All Python source under `src/`, `app/`, and `hf_space/`.
- All tracked notebooks and their stored outputs.
- Raw, processed, generated, and OOD data schemas and distributions.
- Serialized classical models, transformer metadata, label maps, and file sizes.
- Stored metrics, classification reports, leak audit, and split indices.
- Root and Space dependency specifications.
- README, historical planning documents, git history, ignore rules, and deployment workflow.
- Current official regulatory materials listed in Section 14.

### Checks performed

- Parsed every project Python file successfully with `ast.parse`.
- Loaded the deployed classical model and label encoder locally.
- Ran classical inference smoke checks on benign, urgency, empty, and notification strings.
- Verified that the deployed model parameters match `src/train.py` and
  `reports/metrics_summary.json`.
- Checked processed data alignment, missing values, duplicate text, label/category consistency,
  source proportions, class distributions, and OOD coverage.
- Compared duplicate notebooks, duplicate model artifacts, and transformer metadata copies.
- Inspected the local Python environment and package compatibility warnings.

### Not performed

- Full retraining was not run; it is expensive, would rewrite derived artifacts, and the audit
  is about the committed/current state.
- The two hosted apps were not exercised end-to-end from this environment.
- Scraping was not rerun against live third-party sites.
- This is a technical and product-risk review, not a formal legal opinion.

## 3. Current-State Inventory

| Area | Current state | Audit interpretation |
|---|---:|---|
| Processed rows | 6,373 | Internally aligned between `ccpa_dataset.tsv` and `features.csv` |
| Labels | 14 | 13 project dark-pattern labels plus benign |
| Generator rows | 4,217 (66.17%) | Dominant source, not merely augmentation |
| Raw/remapped rows retained | 2,156 (33.83%) | Heuristically remapped from an academic taxonomy |
| Grouped outer split | 5,134 train / 1,239 test | No exact skeleton group crosses this split |
| Naive split contamination | 826/1,275 test rows (64.8%) | Correctly demonstrates random-split inflation |
| OOD benchmark | 23 rows, 8 labels, 0 benign | Too small and incomplete for a release claim |
| Streamlit model | XGBoost, 8.1 MB | Current deployed classical artifact |
| Stored Streamlit-model test result | 0.503 macro-F1 | `reports/metrics_summary.json` |
| Notebook classical results | SVC 0.576, XGB 0.562 | Different pipelines/artifacts from deployment |
| Notebook transformer results | 0.869 grouped test, 0.778 OOD | Development evidence, not an independent final estimate |
| Tests | None | No regression or release protection |
| CI | None | HF sync is deployment automation, not CI |
| License | HF metadata says MIT; no root license file | Distribution terms are ambiguous |

### Source composition by class

| Class | Generated | Raw/remapped | Generated share |
|---|---:|---:|---:|
| Bait and Switch | 320 | 0 | 100.0% |
| Basket Sneaking | 320 | 6 | 98.2% |
| Confirm Shaming | 240 | 75 | 76.2% |
| Disguised Advertisement | 167 | 309 | 35.1% |
| Drip Pricing | 320 | 6 | 98.2% |
| False Urgency | 0 | 493 | 0.0% |
| Forced Action | 320 | 4 | 98.8% |
| Interface Interference | 320 | 63 | 83.6% |
| Nagging | 320 | 0 | 100.0% |
| Not a Dark Pattern | 700 | 1,155 | 37.7% |
| Rogue Malware | 320 | 0 | 100.0% |
| SaaS Billing | 320 | 0 | 100.0% |
| Subscription Trap | 320 | 25 | 92.8% |
| Trick Question | 229 | 21 | 91.6% |

### OOD class coverage

The 23 rows cover Basket Sneaking (2), Disguised Advertisement (2), Drip Pricing (8),
False Urgency (6), Forced Action (1), Nagging (1), SaaS Billing (2), and Subscription Trap
(1). Six labels, including benign, are absent. One error changes accuracy by 4.35 percentage
points. Per-class and macro metrics are therefore highly unstable.

## 4. Severity Model

| Priority | Meaning |
|---|---|
| P0 | Invalidates a core claim, benchmark, release, or user-facing interpretation |
| P1 | Material reliability, reproducibility, security, or maintainability risk |
| P2 | Important quality or scale improvement after trust-critical work |
| P3 | Polish with limited effect on validity or operation |

## 5. P0 Findings

### P0-1. A text-only classifier cannot support the current legal verdict

**Evidence**

- `app/app.py:397-418` renders "CCPA VIOLATION DETECTED" or "CCPA SAFE / BENIGN".
- `hf_space/app.py:127-137` makes the same binary legal-style declaration.
- `app/app.py:31-45` and `hf_space/app.py:45-60` call the 13 categories "Clause 1" through
  "Clause 13". They are specified patterns in Annexure 1, not 13 numbered operative clauses.
- The official category is "Trick Wording" in current government summaries, while the project
  trains and serves "Trick Question".
- Basket Sneaking requires knowing that an item or charge was added without consent. Drip
  Pricing requires price disclosure timing. Nagging requires repetition. Interface Interference
  requires visual/contextual presentation. Subscription Trap requires an enrollment/cancellation
  flow. A standalone string does not contain that evidence.

**Impact**

- False assurance from a "safe" result and false accusation from a "violation" result.
- Legal and reputational risk, especially where generated examples use real company names.
- Product evaluation measures category-language similarity, not actual compliance auditing.

**Required action**

- Replace verdict language with "potential dark-pattern signal" and "no textual signal found".
- Add an explicit abstention state: "insufficient context to assess".
- Rename and version the taxonomy from one canonical legal taxonomy file.
- Collect contextual fields: element type, surrounding copy, default checked state, visual
  prominence, preceding/following price, action required, repetition count, and cancellation
  steps.
- Require human review before a legal/compliance conclusion.

### P0-2. Ground truth is dominated by synthetic and heuristic labels

**Evidence**

- `src/collect_data.py:1-17` describes manually collected examples, but the implementation is a
  random template generator with target counts at `src/collect_data.py:25-27` and generators at
  `src/collect_data.py:143-575`.
- `src.collect_data.build_collected()` reproduces all 4,217 rows in
  `data/processed/collected.tsv`; `data/raw/pattern_label.csv` is the same pool apart from its
  saved index. The checked-in second dataset therefore does not support a distinct Kaggle-source
  claim.
- Generated rows make up 66.17% of the processed corpus. Four classes are entirely generated.
- `src/build_dataset.py:30-60` maps a different academic taxonomy into legal categories using
  string heuristics. For example, every academic Social Proof row becomes Disguised
  Advertisement (`src/build_dataset.py:37-38`), and a question mark can turn a Misdirection row
  into Trick Question (`src/build_dataset.py:44-46`). Unknown categories default to benign.
- `src/build_dataset.py:74` silently skips malformed raw rows.
- `src/build_dataset.py:104` resolves normalized duplicates with `keep="first"` without auditing
  conflicting labels.
- No provenance, annotation confidence, annotator identity, adjudication, or agreement score is
  retained in the final four-column dataset.

**Impact**

- The models can learn generator authorship, template vocabulary, and remapping rules rather
  than real dark-pattern behavior.
- Reported per-class performance is not evidence of legal-category validity.
- Systematic label noise particularly affects Disguised Advertisement, Trick Wording, Interface
  Interference, Subscription Trap, and benign.

**Required action**

- Treat synthetic data as augmentation only and exclude it from validation/test metrics.
- Add a row-level schema with `source_type`, `source_id`, `captured_at`, `jurisdiction`,
  `license`, `is_synthetic`, `template_id`, `annotators`, `adjudicated_label`, `context`, and
  `evidence`.
- Create legal annotation guidelines with positive, negative, and "insufficient context"
  examples.
- Double-annotate real examples and adjudicate disagreements. Report agreement.
- Replace real brands in invented examples with fictional brands.
- Audit rather than silently skip malformed or label-conflicting rows.

### P0-3. Inner validation is still template-leaky

**Evidence**

- The outer split produced by `src/leak_audit.py` is group-aware.
- It is group-aware only for the current skeleton definition. Of 412 real-origin outer-test rows,
  227 share a `page_id` with training, across 111 shared pages.
- `src/train.py:118` then creates ordinary `StratifiedKFold` folds and uses them for model
  comparison (`src/train.py:149-153`) and Optuna (`src/train.py:159-177`).
- Notebook 2 also uses an ordinary random outer split and ordinary stratified CV.
- Stored CV macro-F1 values are 0.943-0.966, while the same training run's grouped holdout
  XGBoost macro-F1 is 0.503.
- `reports/leak_audit.json` independently shows that 64.8% of a naive test split has a template
  twin in training.

**Impact**

- Hyperparameters and model family are selected using a validation signal dominated by
  template recognition.
- The grouped holdout remains useful as a one-time evaluation, but the selected model is not
  optimized for grouped generalization.
- Claims that tuning is "leakage-safe" are incomplete.

**Required action**

- Pass skeleton/template/source groups into every inner fold using `StratifiedGroupKFold`.
- Use nested grouped CV for model selection, or a grouped train/development split plus one
  locked test set.
- Assert no group overlap for outer and inner folds in automated tests.
- Report mean, standard deviation, per-fold class support, and bootstrap confidence intervals.

### P0-4. The OOD benchmark has been used as a development target

**Evidence**

- `src/collect_data.py:83-93` states that generated fee names were grounded in the live examples
  that also populate the OOD evaluation.
- Historical plan files document generator changes intended to improve OOD performance and a
  DistilBERT regularization experiment chosen after observing OOD behavior.
- The blocklist at `src/collect_data.py:30-66` prevents some direct string collisions, but it
  does not undo semantic/vocabulary adaptation to the benchmark.
- The benchmark has only 23 rows, 8 classes, and no benign class.
- All 23 final OOD labels equal the scraper's `auto_label`; only 12 source candidate rows contain
  a human `verified_label`.
- `Only 3 left` overlaps a training row after case normalization, so even exact independence is
  not true.

**Impact**

- The 0.778 transformer OOD macro-F1 is a development result, not an unbiased external estimate.
- Repeated iteration can overfit a tiny set without exact row leakage.

**Required action**

- Rename the current set `ood_dev.csv` and keep it for diagnostics.
- Commission a new, versioned, locked real-world test set after the pipeline is frozen.
- Hide locked labels from model developers or evaluate through a one-way script/service.
- Do not change data, prompts, thresholds, or models based on locked-test errors.

### P0-5. The deployed artifact and reported model are different

**Evidence**

- `app/app.py:53` loads `models/best_multi_model.joblib`.
- That artifact is a 354-tree XGBoost using TF-IDF 1-2 grams. Its parameters match
  `reports/metrics_summary.json` and `src/train.py`, whose grouped holdout macro-F1 is 0.503.
- README reports XGBoost 0.562 and SVC 0.576 from Notebook 3.
- The deployed file is approximately 8.1 MB; README states the classical model is approximately
  2 MB.
- `src/train.py` reports Random Forest as the best CV model (0.966) but always tunes/exports
  XGBoost (`src/train.py:156-187`).
- `models/` and `src/` contain different joblib artifacts with different hashes and parameters.
- `src/train.py`, Notebook 2, Notebook 3, and `src/leak_audit.py` use different n-gram ranges,
  model parameters, split behavior, or result paths.

**Impact**

- Users cannot connect a UI prediction to a truthful model card or metric.
- Retraining from different documented entry points produces different artifacts and numbers.
- Regression detection is impossible without knowing which pipeline is canonical.

**Required action**

- Select one canonical, script-based training entry point.
- Give every artifact a manifest containing data hash, split hash, git SHA, environment lock
  hash, model parameters, calibration parameters, and metrics.
- Make the app display the artifact version and only metrics from that manifest.
- Archive or remove duplicate joblibs and duplicate Notebook 2 after preserving history.
- Make notebooks consume canonical outputs; they must not independently export production models.

### P0-6. Runtime specifications cannot reliably load/reproduce the artifacts

**Evidence**

- The deployed joblibs under `models/` contain scikit-learn 1.7.2 objects, while the stale
  joblibs under `src/` contain 1.8.0 objects and the old local baseline contains 1.6.1 objects.
- `requirements.txt` pins scikit-learn 1.6.1 and incorrectly comments that the joblibs were
  created with 1.6.1.
- Local scikit-learn 1.7.2 loads the model with multiple `InconsistentVersionWarning` messages.
- `models/distilbert_darkpattern/config.json` records Transformers 5.12.1.
- `hf_space/requirements.txt` requires `transformers>=4.40,<5.0`.
- Root requirements omit Torch, Transformers, Jupyter, and the scraping dependency even though
  README says one root install reproduces all notebooks.
- Most dependencies have broad lower bounds and there is no lockfile or supported Python
  version for the root project.

**Impact**

- Fresh deployments can fail to deserialize, change behavior, or be impossible to reproduce.
- A successful local smoke check does not establish compatibility with Streamlit Cloud or the
  HF Space image.

**Required action**

- Choose one supported Python version, preferably 3.12 unless deployment evidence supports
  another version.
- Create exact, tested locks for classical serving, transformer serving, training, scraping,
  notebooks, and development.
- Re-export artifacts inside the locked environment.
- Test artifact load and golden predictions in clean CI containers.
- Never rely on cross-version pickle compatibility as a release strategy.

### P0-7. Confidence, fallback, and "calibration" are mathematically invalid

**Evidence**

- `app/app.py:175-190` applies a hard-coded 0.65 threshold and transforms probability with
  `p - 0.16*p^2`.
- When a dark prediction is forced to benign, `app/app.py:184-187` uses `1 - top_dark_probability`
  as benign confidence. That is the probability mass of all other classes, not the benign
  probability.
- `hf_space/app.py:123-131` can force a dark top class to benign and then describe the dark
  top-class probability as confidence in a safe verdict.
- Classical and transformer thresholds differ (0.65 vs 0.50), have no stored selection method,
  and are absent from reported test metrics.
- XGBoost softmax output and DistilBERT softmax output are not calibrated by default.

**Impact**

- Confidence values can mislead users and cannot be compared across models.
- Deployed decision behavior is not represented by the published metrics.

**Required action**

- Remove probability damping immediately.
- Introduce a grouped calibration set and evaluate temperature scaling, isotonic regression,
  or Platt calibration as appropriate.
- Select thresholds from an explicit cost matrix and publish precision/recall/coverage curves.
- Prefer abstention over silently converting uncertain dark predictions to benign.
- Evaluate the exact deployed post-processing path end-to-end.

## 6. P1 Findings

### P1-1. Split integrity relies on positional indices and a weak staleness check

`src/train.py:63-84` checks total index count and maximum index. A reordered dataset of the same
length can pass. The code does not explicitly assert uniqueness, no overlap, full set equality,
valid bounds, class coverage, or a content hash. Store stable row IDs, dataset SHA-256,
grouping-code version, group IDs, and split metadata; validate all invariants before training.

### P1-2. Skeleton grouping is tailored to the generator

`src/leak_audit.py` masks known slot vocab imported from the generator. This is useful for known
templates but does not catch paraphrases, unseen slot terms, copied source pages, or semantic
near-duplicates. Add source/page/domain grouping, normalized edit similarity, MinHash, and an
embedding-neighbor audit. Review nearest cross-split pairs manually.

### P1-3. The canonical data flow is unclear and divergent

- README tells users to execute notebooks.
- Notebook 1 reads `data/raw/pattern_label.csv`, duplicates feature logic, and writes processed
  outputs differently from `src/build_dataset.py` plus `src/make_features.py`.
- Notebook 2 uses a naive random split and can export production joblibs.
- `src/02_model_tuning_export.ipynb` is a second, different Notebook 2.
- `data/raw/pattern_label.csv` and `data/processed/collected.tsv` contain substantially the same
  generated pool in different formats.
- `colab/` and `models/distilbert_darkpattern/` duplicate transformer metadata locally.

Make package commands canonical. Notebooks should load immutable outputs and explain results.

### P1-4. Feature extraction is fragile at startup and limited to English

- `src/features.py:31-38` attempts NLTK downloads at runtime and suppresses every exception.
  Feature extraction later fails if resources are absent.
- Every Streamlit cold start can invoke this bootstrap.
- Tokenization, lemmatization, POS tags, TextBlob sentiment, and ASCII-only cleaning are English
  specific. `clean_and_lemmatize` removes digits, currency, and non-ASCII text.
- `all_caps_ratio` divides uppercase characters by all characters, not alphabetic characters.
- Regexes are recompiled per call and several lack word boundaries.

Vendor/download language assets at build time, fail with an actionable health error, declare
English-only scope, add multilingual tests, and measure whether the 22 features help through an
ablation. The current keyword-only probe (0.614 in the leak audit) already suggests many numeric
features may add little.

### P1-5. No tests protect data, training, inference, or apps

There is no `tests/` directory, test runner configuration, coverage target, or CI workflow. Add:

- Unit tests for taxonomy mapping, normalization, feature extraction, post-processing, and empty
  or Unicode input.
- Data-contract tests for schema, provenance, duplicates, label conflicts, class coverage, and
  split invariants.
- Golden-prediction tests tied to an artifact manifest.
- Integration tests for model load plus the shared inference API.
- App smoke tests with model/network dependencies mocked.
- A small deterministic training test on a fixture dataset.

### P1-6. Deployment happens without validation or immutable revisions

`.github/workflows/sync-hf-space.yml` uploads directly after an `hf_space/**` push. It does not
run tests, validate model compatibility, pin the HF model revision, use an environment approval,
set concurrency, produce an artifact manifest, or provide automated rollback. GitHub Actions are
pinned to mutable major tags rather than commit SHAs. Add CI before deployment and promote a
tested, immutable release to each surface.

### P1-7. HF inference has import-time network and validation risks

`hf_space/app.py:80-103` downloads and loads the model at module import. This makes unit tests,
offline development, health reporting, and graceful retry difficult. The app does not verify
that label count/order matches model logits, and the model config contains generic `LABEL_n`
metadata. Pin `revision`, put real `id2label/label2id` in config, validate shapes/hashes, load
through an explicit cached lifecycle, and expose readiness separately from liveness.

### P1-8. Security, privacy, and supply-chain controls are absent

- Joblib/pickle deserialization can execute code; only trusted, hash-verified artifacts should
  be loaded.
- Dependencies are not locked or hash-pinned; no dependency or secret scanning is configured.
- Hosted apps have no documented input retention/privacy policy.
- Scraping uses a persistent Chrome profile outside the repository but provides no data-handling
  policy for sessions, screenshots, third-party terms, or PII.
- External Google Fonts add a network/privacy dependency to Streamlit.

Add artifact signatures/hashes, least-privilege tokens, Dependabot/Renovate, CodeQL, secret
scanning, a privacy notice, a retention policy, PII redaction, and documented scraping controls.

### P1-9. Dataset and model licensing/provenance are incomplete

The repository has no root `LICENSE`, dataset card, model card, citation file, or per-source
license field. HF metadata claims MIT, but that does not establish permission to redistribute
third-party data or model weights. Add legal attribution, upstream licenses, collection dates,
intended use, prohibited use, limitations, and takedown/contact procedures.

### P1-10. Regulatory documentation is incomplete and partly stale

The project cites the 2023 guidelines but omits the CCPA's June 2025 self-audit advisory and
subsequent enforcement context. The official current terminology should be represented exactly,
with a `taxonomy_version` and source URL. Product copy should distinguish an ML signal from an
official finding, unfair-trade-practice analysis, or legal violation.

### P1-11. UX and accessibility are not tested

The Streamlit app injects a dark-theme-specific CSS layer, imports external fonts, collapses
input labels, uses extensive emoji, and has no documented responsive/accessibility testing.
Result text is not solely color-dependent, which is good, but contrast, keyboard order, screen
reader output, reduced motion, zoom, and mobile layout are unverified. Use semantic components,
visible labels, accessible result announcements, theme tokens, and automated plus manual WCAG
checks.

## 7. P2 and P3 Findings

### P2-1. No production observability or model monitoring

There are no structured logs, latency/error metrics, model-version counters, drift monitoring,
feedback capture, or safe rollback indicators. If telemetry is added, it must be opt-in and avoid
storing submitted copy by default.

### P2-2. No batch or workflow-oriented audit experience

The apps classify one string at a time. A useful screening tool should accept a page audit
record or CSV, retain context per element, group results by user flow, allow reviewer correction,
and export an evidence report. Build this only after taxonomy and inference are trustworthy.

### P2-3. Training is monolithic and configuration is scattered

Constants, paths, labels, thresholds, feature lists, and model parameters are duplicated across
scripts and notebooks. `src/train.py` mixes comparison, tuning, evaluation, plotting, binary
training, full-data refit, and serialization. Split these into configuration, data, evaluation,
training, calibration, registry, and CLI modules.

### P2-4. Error handling and logging are too broad

Several places catch `Exception` and silently continue, including NLTK downloads and sample
loading. Scripts use `print` rather than structured logging. Catch expected errors, preserve
causes, and fail closed for missing model/data dependencies.

### P2-5. Model strategy has not been justified by clean ablations

SMOTE interpolates sparse TF-IDF and engineered values into points that may not represent valid
text. The feature set, class weights, SMOTE, word n-grams, and model choice need grouped ablation.
Compare at least character TF-IDF, word TF-IDF, class weighting, no-SMOTE, calibrated linear
models, XGBoost, and DistilBERT under identical splits.

### P3-1. Git hygiene hides important work

`.gitignore` ignores every Markdown file except README and ignores all reports. Existing planning
and audit documents therefore do not appear in normal git review, and this report is ignored as
well. Track intentional docs and immutable release reports; ignore only transient outputs.

### P3-2. Repository debris and duplicates increase confusion

Local `.DS_Store`, `.baseline_4391`, `.playwright-mcp`, duplicate transformer directories,
duplicate joblibs under `src/`, and a duplicate Notebook 2 obscure the source of truth. Archive
historical material outside runtime paths and keep only one authoritative artifact location.

## 8. Positive Findings to Preserve

1. The repository openly demonstrates random-split leakage rather than hiding the lower grouped
   result.
2. The current outer split has zero exact skeleton groups crossing train and test and covers all
   14 classes.
3. Classical training and Streamlit inference share `src/features.py`, reducing one form of
   train/serve skew.
4. SMOTE is inside the imbalanced-learn pipeline, so it is not fit before a fold is created.
5. The scraper marks automatic labels as suggestions and explicitly requires human review.
6. The OOD collision blocklist is a useful guard, even though it does not make the benchmark
   independent after benchmark-driven development.
7. The two deployment surfaces avoid forcing transformer dependencies into the lightweight app.
8. Stored notebook outputs and reports expose per-class failures, including zero-F1 classes,
   rather than publishing accuracy alone.
9. Both apps include a research/not-legal-advice disclaimer.
10. Core scripts generally use fixed random seeds and repository-relative paths.

## 9. Target Architecture

The goal is one canonical package, immutable data/model releases, and thin apps.

```text
dark-pattern-detector/
|-- pyproject.toml
|-- uv.lock                         # or another exact lock selected by the team
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- Makefile                        # thin aliases over package commands
|-- configs/
|   |-- taxonomy_ccpa_2023.yaml
|   |-- classical.yaml
|   `-- transformer.yaml
|-- data/
|   |-- README.md                   # dataset card and acquisition policy
|   |-- raw/                        # immutable, checksummed inputs
|   |-- interim/                    # generated outputs, not hand-edited
|   |-- processed/                  # versioned release data
|   `-- manifests/                  # hashes, provenance, schema versions
|-- src/dark_pattern_detector/
|   |-- taxonomy.py
|   |-- schemas.py
|   |-- data/
|   |   |-- build.py
|   |   |-- validate.py
|   |   `-- split.py
|   |-- features/
|   |   |-- classical.py
|   |   `-- context.py
|   |-- models/
|   |   |-- train_classical.py
|   |   |-- train_transformer.py
|   |   |-- calibrate.py
|   |   `-- registry.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- bootstrap.py
|   |   `-- report.py
|   |-- inference.py                # only public inference contract
|   `-- cli.py
|-- apps/
|   |-- streamlit_app.py
|   `-- hf_space_app.py
|-- notebooks/                      # narrative only; no production exports
|-- models/
|   `-- manifests/                  # model binaries live in versioned registry/HF
|-- reports/releases/<version>/     # immutable evaluated release reports
|-- tests/
|   |-- unit/
|   |-- data/
|   |-- integration/
|   `-- smoke/
`-- .github/workflows/
    |-- ci.yml
    |-- data-validation.yml
    `-- deploy.yml
```

### Canonical inference contract

Both apps should call one function returning a versioned object rather than implementing their
own threshold logic:

```python
Prediction(
    taxonomy_version="ccpa-dark-patterns-2023",
    model_version="...",
    status="signal" | "no_textual_signal" | "insufficient_context" | "abstain",
    category="False Urgency" | None,
    calibrated_probability=0.0,
    evidence=[...],
    required_context=[...],
    limitations=[...],
)
```

## 9A. Classical Model Improvement Track

**Implemented 2026-07-15.** The canonical split now uses connected page/template groups. The
initial character-only candidate selected 2-6 grams with `C=1.0` at `0.776 ± 0.030`
training-only macro-F1. It scored `0.733` macro-F1 / `0.818` accuracy on the new 1,322-row
source-clean test and `0.533` macro-F1 on OOD development data; it remains the stronger ablation.

**Final user-selected classical variant:** character 2-6 gram TF-IDF, 12 focused engineered
features, SMOTE, LinearSVC (`C=0.5`), and three-fold grouped sigmoid calibration. It scored
`0.739 ± 0.059` grouped CV, `0.732` test macro-F1 / `0.825` accuracy, and `0.517` OOD-dev
macro-F1. The stronger character-only ablation remains documented rather than hidden.

**Notebook sync completed 2026-07-15:** all notebooks now contain their code directly rather
than importing project scripts. Both Notebook 2 copies are byte-identical, and Notebook 2 and
Notebook 3 share the same embedded classical definition. All old random-split/XGBoost export
code is gone. The user reran the full comparison on the page/template-grouped split:
DistilBERT reached `0.883` test / `0.697` OOD-dev macro-F1, the independently refit classical
row reached `0.730` / `0.517`, and the LSTM reached `0.657` / `0.389`.

The old advanced components were then ablated on the same grouped training folds. Adding the 22
engineered features scored `0.739` after convergence; adding SMOTE scored `0.739`; the legacy SVC scored `0.555`;
and the legacy XGBoost scored `0.559`. Three fresh group-aware Optuna XGBoost trials reached only
`0.566`, `0.576`, and `0.549`, so the search was stopped rather than tuning the weaker family.

Yes, the classical model can probably improve without more synthetic data or a more complex
classifier. A one-off experiment on the existing 5,134-row training partition used five-fold
`StratifiedGroupKFold` and did not score or tune against the outer test.

| Training-only grouped CV candidate | Macro-F1 mean | Std |
|---|---:|---:|
| Current engineered features + word TF-IDF + SMOTE + LinearSVC | 0.533 | 0.080 |
| Raw-text character TF-IDF + class-weighted LinearSVC | **0.773** | 0.069 |

The folds used connected groups: rows stayed together when they shared either a template skeleton
or a `page_id`. A skeleton-only version produced the same conclusion (0.767 versus 0.561). These
are development estimates, not new project headline scores.

### Candidate to carry forward

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=30_000, sublinear_tf=True,
    )),
    ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
])
```

Why this is a good fit: character n-grams preserve numbers, currency, punctuation, spelling
variants, and short fragments that the current ASCII cleanup/lemmatization weakens. LinearSVC is
also a more natural match for sparse TF-IDF than tree boosting plus synthetic SMOTE points.

### Minimal experiment plan

1. Make the connected template/page groups canonical and verify zero overlap in every fold.
2. Compare only six character-SVC settings: `C` in `{0.5, 1, 2}` and n-grams `(3, 5)` or
   `(2, 6)`. Select by training-only grouped macro-F1 and per-class stability.
3. Compare the winner against word-only SVC and the current XGBoost pipeline. Keep engineered
   features only if a grouped ablation proves they help.
4. Freeze preprocessing and parameters, then evaluate once on a newly source-clean outer test.
5. If the UI needs confidence, calibrate on a separate grouped validation partition and add an
   abstention band. Do not convert a low-confidence dark prediction into benign.

Do not spend time on a larger XGBoost search, more keyword flags, ensembling, or more templates
aimed at the current 23-row OOD set. Label quality and source-independent evaluation are the
ceiling; the character SVC is the smallest credible modeling improvement.

## 10. Phased Plan of Action

Estimates assume one engineer with periodic data-annotation and legal-domain support. They are
engineering estimates, not calendar commitments.

### Phase 0: Contain misleading output and freeze the baseline (1-2 days)

**Tasks**

- Freeze current data, split, artifacts, reports, and hashes in a `legacy-v1` manifest.
- Change both apps from "violation/safe" to "potential textual signal/no textual signal".
- Add "insufficient context" guidance by category.
- Remove transformed confidence and misleading fallback confidence.
- Make README identify the actual deployed XGBoost and its 0.503 stored grouped-holdout result.
- Mark 0.778 OOD as development-set performance.
- Stop production export from notebooks.

**Exit criteria**

- No UI declares legal compliance or violation from one string.
- Every displayed metric maps to the loaded artifact manifest.
- The legacy state can be reproduced or at least forensically identified by hashes.

### Phase 1: Reproducible environment and CI foundation (3-5 days)

**Tasks**

- Introduce `pyproject.toml`, one supported Python version, and exact locks.
- Separate dependency groups: `classical-app`, `transformer-app`, `training`, `scraping`,
  `notebooks`, and `dev`.
- Re-export models in those exact environments.
- Add Ruff, a formatter, mypy or Pyright for the core package, pytest, and coverage.
- Add CI for lint, types, unit tests, data-contract tests, artifact load, golden inference, and
  app import/smoke checks.
- Gate HF/Streamlit deployment on CI and immutable release approval.

**Exit criteria**

- A clean Linux runner installs from lock and loads each artifact without version warnings.
- CI runs on pull requests and `main`.
- Deployment cannot run when tests fail.

### Phase 2: Taxonomy and data rebuild (3-6 weeks; parallel annotation work)

**Tasks**

- Have a qualified reviewer validate exact taxonomy names, definitions, applicability, and
  contextual evidence requirements against official sources.
- Publish annotation guidelines and an "insufficient context" label.
- Build a provenance-rich schema and migrate current rows without pretending missing metadata
  exists.
- Separate real, remapped, and generated datasets physically and logically.
- Audit licenses and remove data that cannot be redistributed.
- Collect real examples across every class plus hard benign negatives.
- Double-annotate, adjudicate, and report agreement by class.
- Keep generated data in training-only augmentation experiments.
- Create `ood_dev_v1` from the current 23 rows and a new locked real test after development is
  frozen.

**Suggested benchmark target**

- Minimum 75-100 adjudicated real examples per class, plus at least 300 diverse benign/hard
  negative examples.
- Multiple domains, sites, time periods, devices, and UI-flow contexts.
- Source/site/time grouping so the same campaign or interface cannot cross splits.

**Exit criteria**

- Every evaluated row has provenance, license status, context, and adjudicated label.
- Synthetic rows are zero percent of validation and test.
- All 14 outcomes, including benign, have meaningful locked-test support.

### Phase 3: Evaluation redesign (1-2 weeks)

**Tasks**

- Implement group-aware outer and inner splits using stable group IDs.
- Store split manifests with row IDs, dataset hash, class support, group support, and algorithm
  version.
- Establish three sets: training, visible development, and locked final test.
- Add baselines: majority, keyword rules, word TF-IDF, character TF-IDF, and calibrated linear
  classifier.
- Run grouped ablations for engineered features, SMOTE, class weights, and text normalization.
- Report per-class precision/recall/F1/support, macro-F1, binary metrics, confusion matrices,
  calibration error, Brier score, risk-coverage, latency, memory, and bootstrap intervals.
- Evaluate exact deployed thresholds and abstention.
- Define a cost matrix with domain reviewers rather than choosing a confidence constant.

**Exit criteria**

- No group overlaps in any fold, enforced by tests.
- Model selection uses development evidence only.
- Final-test evaluation is a one-time versioned release operation.

### Phase 4: Model and calibration work (1-3 weeks)

**Tasks**

- Choose the simplest model that wins under clean grouped evaluation.
- Start with calibrated character+word linear baselines; short UI strings often benefit from
  character features and they provide a strong leakage-resistant benchmark.
- Retrain the transformer only after data and splits are frozen.
- Evaluate a two-stage design: textual-risk/benign/insufficient-context first, then category.
- Add explicit abstention and calibration.
- For context-dependent classes, add structured context or a multimodal/flow model rather than
  forcing a text classifier to guess.
- Add robustness slices: very short text, long text, Unicode, Hindi/Hinglish if in scope,
  currency, negation, brand substitution, paraphrase, and adversarial benign promotions.
- Publish a model card and artifact manifest for every release candidate.

**Exit criteria**

- The selected model beats simple baselines with confidence intervals on real grouped data.
- Calibration and abstention meet predefined risk targets.
- Each category's unsupported-context behavior is tested.

### Phase 5: Shared inference and product redesign (1-2 weeks)

**Tasks**

- Extract all prediction and post-processing into one import-safe inference package.
- Make Streamlit and Gradio thin adapters over the same typed contract.
- Ask for the context needed by the selected category instead of accepting only text.
- Show evidence, alternative labels, uncertainty, model version, taxonomy version, and limits.
- Add reviewer correction and a local/exportable audit record.
- Add accessible labels, keyboard behavior, screen-reader announcements, responsive tests, and
  light/dark theme support.
- Document privacy and avoid retaining user input by default.

**Exit criteria**

- Both apps return identical results for identical structured inputs and artifact versions.
- Empty, malformed, long, Unicode, and unavailable-model states fail safely.
- Accessibility smoke checks and manual keyboard/screen-reader review pass.

### Phase 6: Release engineering and operations (3-5 days)

**Tasks**

- Pin HF model/Space revisions and validate artifact hashes.
- Add liveness/readiness checks, structured logs, latency/error counters, and release IDs.
- Use protected deployment environments, least-privilege secrets, concurrency cancellation, and
  documented rollback.
- Add dependency, secret, and code scanning.
- Build a release report automatically from evaluated artifacts.
- Add drift/feedback monitoring only with explicit privacy controls.

**Exit criteria**

- A release is immutable, traceable, test-gated, observable, and rollbackable.
- Production surfaces expose the same release ID as the report/model card.

### Phase 7: Documentation and repository cleanup (2-4 days)

**Tasks**

- Add root license, citation file, dataset card, model cards, contribution guide, security
  policy, and data-collection policy.
- Rewrite quickstart around package commands and exact environments.
- Document data lineage and reproduce every release report from one command.
- Archive duplicate notebooks/artifacts and remove runtime files from `src/`.
- Track intentional Markdown documents and immutable release reports in git.

**Exit criteria**

- A fresh contributor can install, validate data, run a small training job, evaluate, and launch
  each app from documented commands.
- Repository structure has one obvious source of truth for every stage.

## 11. Prioritized Backlog

| ID | Priority | Work item | Estimate | Dependency | Acceptance signal |
|---|---|---|---:|---|---|
| A01 | P0 | Remove legal violation/safe verdicts | 0.5 day | None | Both UIs use risk-screening language |
| A02 | P0 | Remove fake confidence and add abstention | 0.5-1 day | None | UI never relabels uncertainty as benign confidence |
| A03 | P0 | Create legacy artifact/data manifest | 1 day | None | Hashes, params, metrics, git SHA captured |
| A04 | P0 | Correct README deployed model/metric claims | 0.5 day | A03 | README metric matches loaded artifact |
| A05 | P0 | Standardize exact runtime locks and re-export | 2-3 days | A03 | Clean-container loads with no warnings |
| A06 | P0 | Replace inner CV with group-aware validation | 1-2 days | Stable group IDs | Tests prove no group overlap |
| A07 | P0 | Reclassify current OOD as development data | 0.5 day | None | No independent-test claim remains |
| A08 | P0 | Define canonical script pipeline | 1-2 days | A05 | One command owns each artifact |
| A09 | P1 | Add unit/data/integration tests and CI | 3-5 days | A05, A08 | Required CI green on PRs |
| A10 | P0 | Validate legal taxonomy and evidence rules | 2-4 days | Domain reviewer | Versioned approved taxonomy |
| A11 | P0 | Introduce provenance-rich dataset schema | 2-3 days | A10 | Every new row satisfies schema |
| A12 | P0 | Collect and adjudicate real benchmark | 3-6 weeks | A10, A11 | Coverage and agreement targets met |
| A13 | P1 | Add split hashes and stable row IDs | 1-2 days | A11 | Reordering cannot pass stale split |
| A14 | P1 | Run grouped baselines and ablations | 3-5 days | A06, A12 | Release report with intervals |
| A15 | P1 | Calibrate models and define cost thresholds | 2-4 days | A12, A14 | Calibration/risk targets met |
| A16 | P1 | Build shared typed inference package | 2-4 days | A08, A15 | Both apps pass parity tests |
| A17 | P1 | Add structured context inputs | 4-8 days | A10, A16 | Context-dependent labels can abstain |
| A18 | P1 | Pin immutable deploy revisions and gate release | 2-3 days | A09, A16 | Tested release promotion only |
| A19 | P1 | Add model/dataset cards and licenses | 2-4 days | A10-A12 | Distribution and limits documented |
| A20 | P2 | Accessibility and responsive pass | 2-3 days | A16 | WCAG checks plus manual review |
| A21 | P2 | Observability, privacy, and rollback | 2-4 days | A18 | Release ID, metrics, rollback tested |
| A22 | P3 | Archive duplicates and clean ignore rules | 1 day | A08 | One authoritative copy remains |

## 12. Test Strategy

### Unit tests

- Exact official taxonomy names and IDs.
- Academic-to-project mapping, including explicit rejection/uncertainty cases.
- Normalization and skeleton behavior on currency, numbers, punctuation, Unicode, and brands.
- Feature extraction on empty, one-token, long, non-English, and malformed values.
- Probability calibration, abstention, and category-specific context requirements.
- Artifact manifest validation and label-order checks.

### Data tests

- Schema and non-null constraints.
- Stable unique row ID and source ID.
- Label/binary consistency.
- Exact duplicate and conflicting-label detection.
- Near-duplicate report with reviewed thresholds.
- No template/source/site/campaign overlap across splits.
- Required per-class real support and no synthetic validation/test rows.
- Dataset and split hashes match manifests.

### Model tests

- Small deterministic training fixture.
- Golden predictions with tolerances, not brittle exact floats where nondeterminism applies.
- Per-class minimum behavior on curated canonical cases and hard negatives.
- Calibration and abstention regression.
- Character/word baseline comparison.
- CPU latency and memory budget.

### Integration and application tests

- Load artifact from a clean locked environment.
- Both apps call the same inference contract and return parity.
- Model unavailable, label-map mismatch, network failure, and corrupt artifact behavior.
- Empty, oversized, Unicode, HTML-like, and multiline input.
- No raw user text in logs by default.
- Keyboard, screen reader, contrast, zoom, and mobile viewport checks.

### Release gates

- All tests and security scans pass.
- Dataset/model/split manifests are immutable and signed or checksum-verified.
- Release report is generated from the exact candidate artifact.
- Human approval confirms taxonomy language and UI claims.
- Staged smoke test passes before production promotion.

## 13. 30/60/90-Day Sequence

### Days 0-30: Make the current project honest and reproducible

- Complete Phase 0 and Phase 1.
- Freeze legacy artifacts and correct claims.
- Establish exact environments, tests, CI, and canonical commands.
- Finalize legal taxonomy and annotation guide.
- Start provenance migration and real-data collection.

**Day-30 outcome:** a safe research demo with traceable artifacts and no misleading legal or
confidence claims.

### Days 31-60: Build the real benchmark

- Continue multi-annotator collection and adjudication.
- Separate generated augmentation from evaluation.
- Implement stable row/group IDs and locked split manifests.
- Run clean baselines and grouped ablations on available real development data.
- Design shared inference and contextual input schema.

**Day-60 outcome:** a defensible real development benchmark, clear evidence requirements, and a
model-selection process that cannot learn from template leakage.

### Days 61-90: Produce a release candidate

- Freeze the locked test set.
- Train/calibrate candidates and perform one final evaluation.
- Complete shared inference, redesigned UIs, model/dataset cards, accessibility, privacy, and
  release automation.
- Deploy a staged immutable release, validate, then promote.

**Day-90 outcome:** a versioned risk-screening release whose claims, metrics, data, model,
post-processing, and hosted artifact all refer to the same tested object.

## 14. Regulatory Sources and Framing

The implementation should link directly to and version against primary official sources:

- [CCPA Guidelines for Prevention and Regulation of Dark Patterns, 2023](https://consumeraffairs.nic.in/sites/default/files/file-uploads/latestnews/central-consumer-protection-authority-dark-patterns-guidelines-watermark-1565354.pdf)
- [Department of Consumer Affairs consumer-protection rules and guidelines index](https://consumeraffairs.nic.in/acts-and-rules/consumer-protection/consumer-protection)
- [June 2025 CCPA self-audit advisory summary](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2134765&lang=2&reg=48)
- [June 2026 CCPA enforcement update](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2268302&lang=1&reg=1)

Technical product copy should say:

> This system screens submitted content for language associated with specified dark patterns.
> A result is not a legal finding. Many categories require visual, behavioral, transactional,
> or repeated-interaction evidence that a text-only model cannot observe.

## 15. Risks to Track

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Real benchmark remains too small | High | High | Fund annotation; reduce taxonomy scope if necessary |
| Label ambiguity remains high | High | High | Context schema, adjudication, insufficient-context label |
| Developers tune on locked test | Medium | High | Hidden labels and one-way release evaluation |
| Artifact/runtime drift recurs | High | High | Exact locks, manifests, clean-container CI |
| Legal claims outpace model evidence | High | High | Human review, approved copy, abstention |
| Synthetic data creates shortcuts | High | High | Training-only flag, grouped ablations, real-only metrics |
| Scraped data violates terms/privacy | Medium | High | Collection policy, review, redaction, license fields |
| Hosted artifact is replaced upstream | Medium | High | Pin revision and verify hash |
| Calibration degrades after retraining | Medium | Medium | Calibration gates and risk-coverage regression |
| Scope expands before data is ready | High | Medium | Gate product work on benchmark exit criteria |

## 16. Definition of Done

The project is ready to call a defensible dark-pattern **risk screener** when all of the
following are true:

- Taxonomy and product language have domain-review approval and exact source/version links.
- The system abstains when text lacks the context required for a category.
- Validation and test data are real, provenance-rich, adjudicated, licensed, and free of
  training-source/template overlap.
- The locked test covers all outcomes with enough support for useful confidence intervals.
- Inner and outer model selection are group-aware and automatically verified.
- Synthetic data is excluded from reported validation/test metrics.
- Confidence is calibrated; threshold and abstention behavior are evaluated as deployed.
- Every app prediction identifies a model, taxonomy, data, split, and release version.
- Dependencies and artifacts load without warnings from exact clean environments.
- CI tests data, training fixtures, inference, apps, security, and accessibility before deploy.
- Releases are immutable, test-gated, observable, privacy-aware, and rollbackable.
- README, dataset card, model card, license, and release report agree with the actual hosted
  artifact.

## 17. Work That Should Not Be Prioritized Yet

- A larger transformer, LLM, RAG layer, or ensemble before the benchmark is fixed.
- More synthetic templates intended to raise the 23-row OOD score.
- Visual polish that preserves legal-verdict language or fake confidence.
- Hyperparameter sweeps on ordinary stratified folds.
- Adding more engineered keywords without a grouped ablation.
- Claiming multilingual support from an English-only preprocessing pipeline.
- Batch compliance reports before contextual evidence and reviewer workflow exist.

The next unit of progress is **better evidence and release integrity**, not a higher score on the
current development benchmark.
