# Customer Churn Prediction — Classification Models

Machine Learning Assignment 2 — five classification models with six evaluation
metrics, served through an interactive Streamlit application.

---

## a. Problem Statement

Telecom operators lose a substantial share of revenue to **customer churn** —
subscribers cancelling their service and moving to a competitor. Acquiring a new
customer costs several times more than retaining an existing one, so identifying
customers *before* they leave allows the business to intervene with a targeted
retention offer.

**Objective.** Given a customer's demographics, the services they subscribe to,
their contract type and their billing history, predict whether that customer will
churn.

This is a **binary classification** problem:

- `Churn = Yes (1)` — the customer left the company
- `Churn = No (0)` — the customer was retained

**Why the metric choice matters.** Roughly 73% of customers do not churn, so a
model that blindly predicts "No" for everyone already scores about **0.74
accuracy** while identifying **zero** churners and delivering no business value.
Accuracy alone is therefore misleading on this dataset, which is why all six
metrics — and particularly **MCC** — are reported.

---

## b. Dataset Description

**Telco Customer Churn** — a widely used public telecom retention dataset
(IBM sample dataset, mirrored on Kaggle and the UCI repository).

| Property | Value |
|---|---|
| Rows (instances) | **7,043** raw · 7,021 after removing 22 duplicates |
| Features | **19** |
| Target | `Churn` — binary (Yes / No) |
| Class balance | 26.5% churn · 73.5% retained (**2.77 : 1**) |
| Missing values | None, after repairing `TotalCharges` |
| Problem type | Binary classification |

Requirement check: **7,043 instances ≥ 500** ✔ and **19 features ≥ 12** ✔

### Feature groups

| Group | Columns |
|---|---|
| **Demographic** (4) | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account** (5) | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` |
| **Phone services** (2) | `PhoneService`, `MultipleLines` |
| **Internet services** (7) | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Billing total** (1) | `TotalCharges` |

4 numeric columns · 15 categorical columns.

### Data quality issue found and fixed

`TotalCharges` is delivered as **text rather than a number**. Eleven rows contain
a blank string, and every one belongs to a customer with `tenure = 0` — they have
been assigned a monthly charge but have not yet accumulated a total. A naive
`read_csv` silently types the whole column as `object`, which would push a
continuous numeric variable through one-hot encoding and destroy its meaning.
These entries are coerced to numeric and set to `0.0`.

### Preprocessing

| Step | Applied to | Reason |
|---|---|---|
| `StandardScaler` | 4 numeric columns | KNN and SVM are distance/margin based and would otherwise be dominated by `TotalCharges`, whose range is ~1000× larger than `tenure` |
| `OneHotEncoder(drop='first')` | 15 categorical columns | Unordered labels; integer encoding would invent a false ordering |
| Stratified 80/20 split | whole dataset | Preserves the 26.5% churn rate in both splits; `random_state=42` |

All preprocessing is wrapped inside each model's `Pipeline`, so the scaler and
encoder are **fitted on training data only** — no test-set leakage.

---

## c. GitHub Repository Link

**Repository:** https://github.com/2026ak05281-svg/MLAssignment2

**Live Streamlit app:** https://ml-assignment2-2026ak05281.streamlit.app/

### Repository structure

```
project-folder/
├── app.py                      Streamlit application (entry point)
├── requirements.txt            Dependencies for Streamlit Cloud
├── README.md                   This file
├── test_data.csv               Held-out test set (1,405 rows, features + target)
├── train_models.py             Source code — trains and saves all 5 models
├── model/                      Saved models + preprocessing artifacts + source
│   ├── logistic_regression.joblib        trained model 1
│   ├── decision_tree.joblib              trained model 2
│   ├── k_nearest_neighbors.joblib        trained model 3
│   ├── naive_bayes_gaussian.joblib       trained model 4
│   ├── support_vector_machine.joblib     trained model 5
│   ├── preprocessor.joblib               preprocessing artifact (fitted)
│   ├── metrics.json                      metrics, ROC data, input schema
│   ├── train_models.py                   source code that built these artifacts
│   └── README.md                         describes every file above
├── data/
│   └── telco_churn.csv         Full dataset
└── notebook/
    └── Telco_Churn_Classification.ipynb   Source code (.ipynb) with full analysis
```

### Saved models and preprocessing artifacts

Each `.joblib` model file stores a complete `Pipeline` — the fitted
`StandardScaler`, the fitted `OneHotEncoder` and the trained classifier — so the
preprocessing travels with the model and the app never needs to retrain or
reapply any transformation.

`preprocessor.joblib` additionally stores the **standalone fitted
`ColumnTransformer`**, so the scaler statistics and encoder categories can be
inspected or reused on their own. It transforms the 19 raw input features into
**30 encoded columns** (4 scaled numeric + 26 one-hot indicators).

The source code that generated every artifact is included both at the repository
root (`train_models.py`), inside `model/` as an exact copy, and as a notebook
(`notebook/Telco_Churn_Classification.ipynb`).

---

## d. Models Used

All five models were trained on the identical 80/20 stratified split
(`random_state=42`) and evaluated on the same 1,405-row held-out test set.

### Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8021** | **0.8404** | 0.6599 | 0.5215 | **0.5826** | **0.4607** |
| Decision Tree | 0.7851 | 0.8215 | 0.6471 | 0.4140 | 0.5049 | 0.3913 |
| KNN | 0.7822 | 0.8228 | 0.6012 | 0.5269 | 0.5616 | 0.4192 |
| Naive Bayes | 0.6512 | 0.8078 | 0.4234 | **0.8763** | 0.5709 | 0.3959 |
| Support Vector Machine | 0.7972 | 0.7852 | **0.6851** | 0.4328 | 0.5305 | 0.4270 |

Best value per metric shown in **bold**.
Majority-class baseline accuracy: **0.7352**

### Supporting confusion matrices

| Model | TN | FP | FN | TP | Churners caught |
|---|---|---|---|---|---|
| Logistic Regression | 933 | 100 | 178 | 194 | 194 / 372 = 52.2% |
| Decision Tree | 949 | 84 | 218 | 154 | 154 / 372 = 41.4% |
| KNN | 903 | 130 | 176 | 196 | 196 / 372 = 52.7% |
| Naive Bayes | 589 | **444** | **46** | **326** | **326 / 372 = 87.6%** |
| Support Vector Machine | 959 | **74** | 211 | 161 | 161 / 372 = 43.3% |

### Model configurations

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, default L2 penalty |
| Decision Tree | `max_depth=6` — capped to prevent overfitting |
| KNN | `n_neighbors=21` — odd value avoids ties; large k smooths noise |
| Naive Bayes | `GaussianNB`, default settings |
| Support Vector Machine | `SVC(kernel='rbf')` wrapped in `CalibratedClassifierCV` |

> **Note on the SVM.** `SVC` exposes no native `predict_proba`, but **AUC needs a
> continuous score**, not hard labels. It is therefore wrapped in
> `CalibratedClassifierCV(..., ensemble=False)`, which fits a calibration model on
> the SVM decision function. (`SVC(probability=True)` was deprecated in
> scikit-learn 1.9.)

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | **Best overall.** Wins AUC (0.8404), F1 (0.5826) and MCC (0.4607), and is within 0.5 points of the best accuracy. The churn signal is close to linearly separable once categories are one-hot encoded, so the simplest model wins. Its probability outputs are naturally well calibrated, which directly lifts AUC. It also strikes the most even precision/recall balance (0.66 / 0.52), catching 52% of churners while keeping false alarms to 100. |
| **Decision Tree** | **Weakest MCC (0.3913).** Capped at `max_depth=6` to prevent overfitting, and that limit costs it recall — it misses 218 of 372 churners, the worst of any model. It is also the most conservative after SVM, producing only 84 false positives. Its probability estimates are coarse because every sample in a leaf shares one probability, which is why its ROC curve is visibly more angular and its AUC (0.8215) trails Logistic Regression despite comparable accuracy. |
| **KNN** | **Middle of the field on every metric.** Highest recall among the three "balanced" models (0.5269) but the lowest precision of them (0.6012), generating 130 false positives. After one-hot encoding, the feature space expands to ~30 dimensions, and Euclidean distance becomes progressively less discriminating as dimensionality grows — the curse of dimensionality. `k=21` smooths noise but blurs local structure. It is also the slowest at prediction time, since every query is compared against all 5,616 training rows. |
| **Naive Bayes** | **The most extreme and most interesting result.** By far the best recall (**0.8763** — catches 326 of 372 churners) but the worst precision (0.4234) and worst accuracy (0.6512), *below the 0.7352 majority baseline*. The cause is visible in the correlation heatmap: `tenure`, `MonthlyCharges` and `TotalCharges` are strongly correlated (`TotalCharges ≈ tenure × MonthlyCharges`), directly violating the conditional-independence assumption. The model effectively counts the same evidence three times, becomes over-confident, and pushes many borderline customers over the churn threshold — producing 444 false positives. Notably its **AUC is still strong (0.8078)**, showing the probability *ranking* is sound even though the 0.5 threshold is badly calibrated. |
| **Support Vector Machine** | **The precision specialist.** Highest precision (**0.6851**) and fewest false positives (only 74), but low recall (0.4328) — it misses 211 churners. The RBF margin is conservative, flagging churn only where evidence is strong. Paradoxically it has the **lowest AUC (0.7852)** despite the second-best accuracy, because the calibration wrapper's probability estimates rank less cleanly than Logistic Regression's native ones. It is also by far the slowest to train (~2.7 s versus <0.2 s for the others), as SVM training scales roughly quadratically with sample count. |
| **Overall winner** | **Logistic Regression.** It takes the best AUC, F1 and MCC, trains in under 0.1 seconds, and its coefficients remain directly interpretable — a genuine advantage when the business needs to explain *why* a customer was flagged. MCC is the decisive metric here because it incorporates all four confusion-matrix cells and cannot be inflated by the 73.5% majority class the way accuracy can. |

### Additional observations

**1. Accuracy is the least informative metric on this dataset.** Predicting "no
churn" for every customer scores 0.7352 — beating Naive Bayes outright while
identifying not a single churner. Any model must be judged against that baseline,
not against zero.

**2. Precision and recall are in direct tension, and the right balance is a
business decision.** A **false negative** is a churner you failed to flag,
costing the customer's entire remaining lifetime value. A **false positive** is a
loyal customer wrongly flagged, costing one unnecessary retention offer. If
retention offers are cheap relative to losing a customer, Naive Bayes — despite
the worst accuracy — may genuinely be the better production choice, since it
catches 87.6% of churners versus Logistic Regression's 52.2%. If offers are
expensive, SVM's precision is preferable. **No single metric settles this; the
full table does.**

**3. AUC and accuracy can disagree sharply, and both are needed.** Naive Bayes
has the *third-best* AUC (0.8078) but the *worst* accuracy (0.6512). AUC measures
how well a model **ranks** churners above non-churners across all thresholds;
accuracy measures correctness at the single fixed 0.5 cut-off. Naive Bayes ranks
customers well but its probabilities are skewed towards 1, so the default
threshold is wrong. Its accuracy could be improved substantially by tuning the
threshold alone, without touching the model.

**4. Model complexity did not help here.** The simplest model beat the RBF SVM and
the Decision Tree on the metrics that matter. With ~30 encoded features and
clean, linearly-informative predictors such as `Contract` and `tenure`, the extra
flexibility of a kernel method bought nothing and cost 30× the training time.

---

## Streamlit Application

**Live app:** https://ml-assignment2-2026ak05281.streamlit.app/

### Features

| Feature | Where |
|---|---|
| **Dataset upload (CSV)** | Sidebar — upload any test CSV containing the feature columns plus the `Churn` target. A bundled `test_data.csv` loads by default. |
| **Model selection dropdown** | Sidebar — choose any of the five models for the detailed view |
| **Display of evaluation metrics** | *Evaluation Metrics* tab — all six metrics for all five models, computed **live on the uploaded data**, with a heat-mapped table, per-metric charts and CSV download |
| **Confusion matrix & classification report** | *Confusion Matrix & Report* tab — confusion matrix for the selected model with TN/FP/FN/TP breakdown, full per-class classification report, plus matrices for all five models side by side |
| ROC curves | *ROC Curves* tab — all five models overlaid with AUC in the legend |
| Model details | *Model Details* tab — configurations, training-time reference metrics, expected input schema |

The app **loads saved models from `model/` and never retrains**. Uploading a
different CSV recomputes every metric against the already-trained pipelines.

### Uploaded CSV requirements

- All 19 feature columns, named exactly as in `test_data.csv`
- A `Churn` target column — accepts `Yes`/`No` or `1`/`0`
- `TotalCharges` may contain blanks; the app repairs them automatically

---

## Running Locally

```bash
pip install -r requirements.txt
python train_models.py      # optional — regenerates model/ and test_data.csv
streamlit run app.py
```

If `streamlit` is not on your PATH, use `python -m streamlit run app.py`.

---

## Technology Stack

| Purpose | Library |
|---|---|
| Models & metrics | scikit-learn |
| Data handling | pandas, numpy |
| Model persistence | joblib |
| Web application | Streamlit |
| Charts | Plotly (app), matplotlib + seaborn (notebook) |
