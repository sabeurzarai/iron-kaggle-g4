# iron-kaggle-g4
### Retail Sales Forecasting — IronHack Kaggle Competition

---

## Project Overview

Predict daily `sales` for retail stores using historical data (2013–2015).

| | |
|---|---|
| **Dataset** | 640,840 rows × 10 columns |
| **Target** | `sales` (integer) |
| **Best model** | XGBoost |
| **R² Validation** | 84.23% |
| **RMSE** | 1,233 euros |

---

## Results

| Model | R² Train | R² Validation | RMSE | R² Difference |
|---|---|---|---|---|
| **XGBoost** | 84.56% | **84.23%** | **1,233** | 0.0033 |
| Random Forest | 81.70% | 81.08% | 1,351 | 0.0062 |
| Linear Regression | 73.50% | 73.66% | 1,594 | -0.0016 |

XGBoost was selected as the winner: best R² Validation, lowest RMSE, and smallest R² Difference (no overfitting).

---

## Data Quality Issues Found

No missing values. However, 4 structural issues required preprocessing:

### 1. Fake column
`Unnamed: 0` is a row index — dropped immediately, no predictive value.

### 2. Wrong type — date stored as string
A raw string like `"2013-04-18"` is meaningless to a model. Extracted into numeric features:

```python
df['date']       = pd.to_datetime(df['date'])
df['year']       = df['date'].dt.year
df['month']      = df['date'].dt.month
df['day']        = df['date'].dt.day
df['week']       = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df = df.drop(columns=['date'])
```

> REAL_DATA uses `DD/MM/YYYY` format — handled with `dayfirst=True`.

### 3. Categorical data — state_holiday stored as text
Values: `'0'` (no holiday), `'a'` (public), `'b'` (easter), `'c'` (christmas).

Linear Regression requires One-Hot Encoding — label encoding would imply a false numeric order (`c > b > a`).

```python
df['state_holiday'] = df['state_holiday'].astype(str)
df = pd.get_dummies(df, columns=['state_holiday'], prefix='holiday', drop_first=True)
# drop_first=True avoids the dummy variable trap (multicollinearity)
```

| Model | Encoding used |
|---|---|
| Linear Regression | One-Hot Encoding (required) |
| Random Forest | One-Hot Encoding |
| XGBoost | One-Hot Encoding |

### 4. Logical edge cases — closed stores and anomaly

**Closed stores:** `open = 0` always means `sales = 0` (108,824 rows).
These rows were removed from training. At prediction time, closed stores are forced to `sales = 0`.

```
Original rows : 640,840
After removing closed stores : 532,016
```

**Statistical anomaly:** 1 row where `open = 1`, `customers = 5`, `sales = 0`.
Out of 531,986 open days with customers, this is the only zero-sales day (0.0002%).
Store 948 normally makes €6,898 average sales when open. This is a data entry error — kept as negligible impact (1 out of 640K rows).

---

## Preprocessing Pipeline

```
1. Drop Unnamed: 0
2. Remove anomalous row (open + customers > 0 + sales = 0)
3. Remove closed stores (open = 0)
4. Convert date string → extract year, month, day, week, is_weekend
5. One-Hot Encode state_holiday (drop_first=True)
6. Align REAL_DATA columns to training feature_columns
7. Force closed stores = 0 at prediction time
```

---

## Repository Structure

```
iron-kaggle-g4/
├── data/
│   ├── training.csv       # 640,840 rows — training data
│   ├── REAL_DATA.csv      # 71,205 rows — test data (no sales column)
│   └── group_4.csv        # Final predictions (submitted)
├── project/
│   └── g4_notebook.ipynb  # Full pipeline: EDA → preprocessing → 3 models → export
├── Presentation_G4_final.pptx
└── README.md
```

---

## How to Run

```bash
# Open the notebook and run all cells top to bottom
jupyter notebook project/g4_notebook.ipynb
```

Outputs:
- Model comparison table printed to console
- `data/group_4.csv` — predictions using the best model (XGBoost)
