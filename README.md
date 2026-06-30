# Credit Default Prediction: Traditional ML vs. Deep Learning

A machine learning pipeline comparing Scikit-learn classifiers against TensorFlow
deep learning models on the task of predicting credit card client default. Built
as a summative project for the Introduction to Machine Learning module.

> **Headline result:** [FILL IN — e.g. "A tuned HistGradientBoosting classifier
> achieved the best ROC-AUC (0.XX), outperforming every neural network
> architecture tested."] This repository documents how that conclusion was
> reached and why it is a reasonable finding for tabular financial data.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Run](#how-to-run)
6. [Methodology](#methodology)
7. [Experiments](#experiments)
8. [Model Training & Hyperparameters](#model-training--hyperparameters)
9. [Optimization Techniques](#optimization-techniques)
10. [Results](#results)
11. [Key Findings](#key-findings)
12. [Dataset Limitations](#dataset-limitations)
13. [Future Work](#future-work)
14. [References](#references)
15. [Author](#author)

---

## Problem Statement

Credit scoring is the mechanism banks use to estimate the probability that a
borrower will fail to meet their obligations. Misclassifying a defaulter as
creditworthy is expensive; misclassifying a good customer as a default risk costs
revenue and goodwill. The task is therefore a **binary classification** problem
with **asymmetric error costs** and a **class imbalance** that makes raw accuracy
a misleading metric.

This project asks a focused question: *on real, tabular credit data, does a deep
neural network actually outperform well-tuned traditional machine learning?* The
pipeline compares Scikit-learn models against TensorFlow models (Sequential API,
Functional API, and the `tf.data` input pipeline) under a controlled,
fair-comparison protocol.

## Dataset

**Default of Credit Card Clients** — UCI Machine Learning Repository.

- **Source:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Instances:** 30,000 clients
- **Features:** 23 (demographics, credit limit, six months of repayment status,
  bill amounts, and payment amounts)
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- **Class balance:** approximately 22% positive (default) — moderately imbalanced
- **Period:** Taiwan, April–September 2005

> The raw data file is **not committed to this repository** (see
> [`.gitignore`](.gitignore)). Download it from the UCI link above and place it in
> `data/` before running the notebook, or run `scripts/download_data.py`.

### Why this dataset
- A ~30k row scale is large enough to justify a neural network but not so large
  that deep learning wins on data volume alone — making the comparison honest.
- A genuine mix of categorical (sex, education, marriage, repayment-status codes)
  and continuous features forces real preprocessing and feature-engineering work.
- Moderate imbalance enables a meaningful discussion of class weighting,
  precision–recall trade-offs, and why accuracy alone is inadequate.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebook.ipynb              # main, runs top-to-bottom (Restart & Run All)
├── data/                       # raw dataset (gitignored) — download separately
├── scripts/
│   └── download_data.py        # fetches the dataset from UCI
├── src/                        # optional: reusable pipeline code
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
├── saved_models/               # best sklearn (.pkl) and TF (.keras) artifacts
├── figures/                    # plots used in the report
└── report/
    └── report.pdf              # academic write-up (figures, IEEE references)
```

## Setup & Installation

```bash
# clone
git clone https://github.com/DUSHIMEDanPaul/<repo-name>.git
cd <repo-name>

# create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# fetch the dataset
python scripts/download_data.py  # or download manually into data/
```

**Core dependencies** (pin your actual versions in `requirements.txt`):

```
python>=3.10
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
jupyter
```

## How to Run

Open the notebook and execute every cell in order:

```bash
jupyter notebook notebook.ipynb
```

The notebook is structured to run cleanly via **Kernel → Restart & Run All**. It
proceeds: data loading → EDA → preprocessing → feature engineering → Scikit-learn
experiments → TensorFlow experiments → evaluation and comparison → conclusions.

## Methodology

### Pipeline overview
1. **Exploratory data analysis** — distributions, missingness, correlations,
   class balance, and inspection of the repayment-status and bill/payment columns.
2. **Preprocessing**
   - Train/validation/test split: **stratified** on the target, `random_state` set
     for reproducibility. *(See hyperparameter rationale below for the split ratio.)*
   - Categorical encoding (one-hot for nominal fields such as education/marriage).
   - Feature scaling (`StandardScaler`) applied **inside** the pipeline and fit on
     the training fold only, to prevent leakage.
3. **Feature engineering** — [FILL IN: your engineered features, e.g. utilization
   ratio = bill / credit limit, average delay across the six months, count of
   months delinquent, payment-to-bill ratios]. Each feature should be motivated by
   domain reasoning, not added blindly.
4. **Traditional ML (Scikit-learn)** — Logistic Regression, Random Forest, and a
   gradient-boosted model (HistGradientBoosting). All wrapped in `Pipeline` /
   `ColumnTransformer` objects so preprocessing is reproducible and leak-free.
5. **Deep learning (TensorFlow)**
   - **Sequential API** — a straightforward feed-forward baseline net.
   - **Functional API** — [FILL IN: e.g. a wide-and-deep architecture, or a model
     with separate input branches for engineered vs. raw features].
   - **`tf.data` pipeline** — input pipeline with batching, shuffling, caching, and
     prefetching for efficient training.
6. **Evaluation** — common metric set across all models on the held-out test set.

### Evaluation metrics
Because the classes are imbalanced, the primary metric is **ROC-AUC** (and/or
**PR-AUC**), reported alongside precision, recall, F1, and a confusion matrix.
Accuracy is reported but explicitly **not** used as the decision metric.

## Experiments

[FILL IN — confirm your real experiment matrix. Example structure below.]

| # | Family       | Model                        | Key variation tested            |
|---|--------------|------------------------------|---------------------------------|
| 1 | Scikit-learn | Logistic Regression          | Baseline, `class_weight=balanced` |
| 2 | Scikit-learn | Random Forest                | Untuned vs. tuned               |
| 3 | Scikit-learn | HistGradientBoosting         | Untuned vs. tuned               |
| 4 | TensorFlow   | Sequential (shallow)         | Baseline net                    |
| 5 | TensorFlow   | Sequential (deeper + dropout)| Regularization effect           |
| 6 | TensorFlow   | Functional API               | Architecture variation          |
| 7 | TensorFlow   | Sequential + class weights   | Imbalance handling              |
| 8 | TensorFlow   | Best net + LR scheduling     | Optimization effect             |

## Model Training & Hyperparameters

This section explains **why** each hyperparameter was chosen, not just what it was
set to. Replace bracketed values with your actual settings and keep the reasoning
that applies.

### Shared protocol
- **Split:** [e.g. 70/15/15] stratified train/val/test. Stratification preserves
  the ~22% default rate in every split so metrics aren't distorted by an unlucky
  draw. A fixed `random_state` makes every result reproducible — essential when
  you claim one model beats another.
- **Scaling:** `StandardScaler` for Logistic Regression and all neural networks,
  because both are sensitive to feature magnitude (gradient steps and the
  regularization penalty scale with feature size). Tree-based models
  (Random Forest, HistGradientBoosting) are scale-invariant, so scaling is
  optional for them and was [kept / omitted] for consistency.

### Logistic Regression
- `class_weight='balanced'` — re-weights the loss so the minority (default) class
  is not ignored; without it the model can score high accuracy by predicting
  "no default" for nearly everyone, which is useless for the business problem.
- `C = [FILL IN]` — inverse regularization strength. Lower `C` = stronger
  regularization. Tuned by [grid/random search] over a log scale (e.g.
  `[0.01, 0.1, 1, 10]`) using cross-validated ROC-AUC. Picked the value that
  maximized validation AUC without a large train/val gap (overfitting check).
- `solver='lbfgs'`, `max_iter=[FILL IN]` — lbfgs is a robust default for this size;
  `max_iter` was raised until the solver converged (no convergence warnings).

### Random Forest
- `n_estimators = [FILL IN]` — more trees reduce variance and stabilize the
  estimate, with diminishing returns; chosen where validation AUC plateaued so the
  extra compute wasn't buying accuracy.
- `max_depth` / `min_samples_leaf = [FILL IN]` — the main overfitting controls.
  Unlimited depth memorizes the training set; constraining depth or enforcing a
  minimum leaf size trades a little training fit for better generalization.
- `class_weight='balanced'` — same imbalance reasoning as above.

### HistGradientBoosting
- `learning_rate = [FILL IN]` — the shrinkage applied to each tree's contribution.
  Smaller values generalize better but need more iterations; this trades off
  directly against `max_iter`.
- `max_iter = [FILL IN]` — number of boosting stages, paired with early stopping
  (`early_stopping=True`, `validation_fraction`, `n_iter_no_change`) so training
  halts once validation loss stops improving instead of overfitting.
- `max_leaf_nodes` / `max_depth = [FILL IN]` — controls per-tree complexity; kept
  modest because boosting builds strength from many weak learners, not few strong
  ones.

### Neural networks (TensorFlow)
- **Architecture:** [e.g. 2–3 hidden layers of 64→32 units]. Tabular data rarely
  benefits from very deep networks — added depth mostly adds overfitting risk — so
  the network is intentionally shallow.
- **Hidden activation:** ReLU — cheap, non-saturating, and the standard default;
  avoids the vanishing-gradient issues of sigmoid/tanh in hidden layers.
- **Output:** single unit with **sigmoid** activation for a probability in [0, 1].
- **Loss:** **binary cross-entropy** — the correct loss for probabilistic binary
  classification.
- **Optimizer:** **Adam**, learning rate `[1e-3 default]` — see optimization
  section for why Adam over plain SGD.
- **Batch size:** `[256 / 512]` — larger batches give smoother gradient estimates
  and faster epochs on tabular data, with little accuracy cost at this scale.
- **Epochs:** `[FILL IN]` upper bound, but training is governed by **early
  stopping** on validation loss/AUC rather than a fixed epoch count.
- **Dropout:** `[0.3–0.5]` — randomly drops units during training to prevent
  co-adaptation and overfitting; tuned by watching the train/val gap.
- **Batch normalization:** [used / not used] — stabilizes and speeds up training
  by normalizing layer inputs; helps the optimizer take larger, safer steps.
- **Class imbalance:** handled via `class_weight` passed to `model.fit()` [and/or
  resampling], so the network is penalized more for missing defaulters.

> **How the values were chosen:** hyperparameters were selected on the
> **validation set only**; the test set was touched once, at the end, to report
> final numbers. Tuning used [grid search / random search / manual search guided by
> validation AUC]. Each reported choice is the configuration that maximized
> validation performance while keeping the train/validation gap small.

## Optimization Techniques

The techniques below are what make training stable and the comparison fair.

- **Adam optimizer.** Combines momentum with per-parameter adaptive learning
  rates, so it converges faster and is far less sensitive to the initial learning
  rate than vanilla SGD — the right default for a from-scratch tabular network.
- **Learning-rate scheduling (`ReduceLROnPlateau`).** When validation loss stops
  improving, the learning rate is cut by a factor, letting the optimizer settle
  into a better minimum instead of bouncing around it.
- **Early stopping.** Monitors validation loss/AUC with a `patience` window and
  restores the best weights, preventing the network from overfitting after it has
  stopped generalizing. This also removes the need to hand-pick an epoch count.
- **Dropout & batch normalization.** Regularization and training-stability tools,
  respectively (see above).
- **Class weighting / resampling.** Counteracts the ~22% imbalance so the models
  optimize for catching defaulters, not just overall accuracy.
- **Feature scaling.** A form of optimization in itself for gradient-based models:
  standardized inputs give the loss surface more uniform curvature, so gradient
  descent converges faster and more reliably.
- **Cross-validation for tuning.** Traditional-ML hyperparameters were selected
  with k-fold cross-validated ROC-AUC, giving a less noisy estimate than a single
  validation split.
- **Fair-comparison controls.** The same splits, the same random seeds, and the
  same evaluation metric were held constant across every model so that performance
  differences reflect the models, not the experimental setup.

## Results

[FILL IN with your real numbers on the held-out test set.]

| Model                    | ROC-AUC | PR-AUC | Precision | Recall | F1   |
|--------------------------|---------|--------|-----------|--------|------|
| Logistic Regression      |  0.XX   |  0.XX  |   0.XX    |  0.XX  | 0.XX |
| Random Forest            |  0.XX   |  0.XX  |   0.XX    |  0.XX  | 0.XX |
| HistGradientBoosting     |  0.XX   |  0.XX  |   0.XX    |  0.XX  | 0.XX |
| Neural Net (Sequential)  |  0.XX   |  0.XX  |   0.XX    |  0.XX  | 0.XX |
| Neural Net (Functional)  |  0.XX   |  0.XX  |   0.XX    |  0.XX  | 0.XX |

Include in the notebook / report: ROC and PR curves, confusion matrices for the
best models, and a training-history plot (loss/AUC vs. epoch) for the networks.

## Key Findings

[FILL IN — base these on your actual results. A typical, defensible narrative:]
- On this tabular dataset, the tuned gradient-boosted model matched or exceeded
  every neural network on ROC-AUC, consistent with the broader literature that
  tree ensembles remain strong baselines for tabular data.
- The networks were more sensitive to preprocessing, scaling, and regularization,
  and required more tuning effort for comparable performance.
- Handling class imbalance changed the recall/precision balance far more than any
  single architecture change.

## Dataset Limitations

- **Single market and period.** Taiwan, 2005 — economic and regulatory context
  limits generalization to other markets or to today.
- **Class imbalance.** ~22% default inflates accuracy and demands imbalance-aware
  metrics and training.
- **Coarse / ambiguous coding.** Some categorical fields (e.g. education, the
  repayment-status scale) contain undocumented or out-of-range codes that require
  judgment calls in cleaning.
- **No cost matrix.** The dataset doesn't encode the real monetary cost of a false
  negative vs. a false positive, so the "best" operating threshold is assumed
  rather than derived.
- **Limited features.** No macroeconomic, behavioral, or time-series signals beyond
  six months of history.

## Future Work

- Threshold optimization driven by an explicit cost matrix.
- Probability calibration (e.g. isotonic / Platt scaling) for usable risk scores.
- SHAP-based interpretability for the boosted model.
- Testing on a second, more recent credit dataset to probe generalization.

## References

[FILL IN — IEEE format, matching your report. Include:]
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default of credit card clients.
- UCI Machine Learning Repository: Default of Credit Card Clients dataset.
- Scikit-learn and TensorFlow documentation.

## Author

**Dan (DUSHIME Dan Paul)**
Software Engineering, African Leadership University
GitHub: [@DUSHIMEDanPaul](https://github.com/DUSHIMEDanPaul)
