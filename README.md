# Heart Disease Prediction Model

A machine learning project that predicts the presence of heart disease from patient clinical data, comparing Logistic Regression, Random Forest, and XGBoost classifiers.

## Dataset

The notebook uses `heart.csv`, containing patient records with the following raw fields:

| Column | Description |
|---|---|
| Age | Patient age in years |
| Sex | M / F |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | Fasting blood sugar (0/1) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | N / Y |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |
| HeartDisease | Target label (0 = no disease, 1 = disease) |

No missing values were found in the dataset (`data.isna().sum()` returned 0 across all columns).

## Feature Engineering

All categorical fields were manually encoded to numeric values so they could be fed into the models:

- **Sex**: `M` → 0, `F` → 1
- **ChestPainType**: `ATA` → 0, `NAP` → 1, `ASY` → 2, `TA` → 3
- **RestingECG**: `Normal` → 0, `ST` → 1, `LVH` → 2
- **ExerciseAngina**: `N` → 0, `Y` → 1
- **ST_Slope**: `Flat` → 0, `Up` → 1, `Down` → 2

**Feature selection via correlation analysis**: After encoding, a correlation matrix was computed against the `HeartDisease` target. Any feature with an absolute correlation greater than **0.1** was kept as a relevant feature. All 11 features (`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`) passed this threshold and were retained.

Two exploratory scatter plots were also produced to visualize relationships in the data:
- Age vs. Max Heart Rate, colored by disease status
- Age vs. Cholesterol, colored by disease status

**Train/test split**: 80/20 split (`test_size=0.2`, `random_state=42`), producing 734 training samples and 184 test samples. No separate validation set was used.

**Scaling**: `StandardScaler` was fit on the training set and applied to both the training and test sets before modeling.

## Models & Results

Three classifiers were trained and evaluated on the same scaled test set.

### 1. Logistic Regression
`LogisticRegression(C=1.0, solver='lbfgs', random_state=42)`

| Metric | Score |
|---|---|
| Accuracy | 0.8207 |
| Precision (class 1) | 0.87 |
| Recall (class 1) | 0.81 |
| F1-score (class 1) | 0.84 |

Confusion Matrix:
```
[[64 13]
 [20 87]]
```

### 2. Random Forest (Grid-Search Tuned)
`RandomForestClassifier` tuned via `GridSearchCV` (5-fold CV) over:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `criterion`: [gini, entropy]

| Metric | Score |
|---|---|
| Accuracy | 0.8750 |
| Precision | 0.8962 |
| F1-Score | 0.8920 |
| ROC-AUC | 0.9307 |

### 3. XGBoost (Grid-Search Tuned)
`XGBClassifier` tuned via `GridSearchCV` (5-fold CV) over:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.01, 0.1, 0.2]
- `subsample`: [0.8, 1.0]

| Metric | Score |
|---|---|
| Accuracy | 0.8750 |
| Precision | 0.8962 |
| F1-Score | 0.8920 |
| ROC-AUC | **0.9359** |

## Summary

| Model | Accuracy | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8207 | 0.87 | 0.84 | — |
| Random Forest | 0.8750 | 0.8962 | 0.8920 | 0.9307 |
| **XGBoost** | **0.8750** | **0.8962** | **0.8920** | **0.9359** |

XGBoost and Random Forest performed identically on accuracy, precision, and F1-score, both outperforming Logistic Regression by roughly 5 percentage points of accuracy. XGBoost edges out Random Forest slightly on ROC-AUC (0.9359 vs 0.9307), making it the best-performing model overall in this notebook.

## Tech Stack

- pandas, matplotlib, seaborn — data handling & visualization
- scikit-learn — preprocessing, Logistic Regression, Random Forest, GridSearchCV, metrics
- xgboost — gradient-boosted tree classifier

## How to Run

1. Place `heart.csv` in the same directory as the notebook.
2. Install dependencies: `pip install pandas matplotlib seaborn scikit-learn xgboost`
3. Run all cells in `main.ipynb` sequentially.
