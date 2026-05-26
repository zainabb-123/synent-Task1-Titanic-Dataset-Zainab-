# Titanic Survival Prediction — Data Preprocessing

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-orange)

---

## Problem Statement

The goal of this project is to preprocess the classic **Titanic dataset** to make it suitable for machine learning models that predict passenger survival. The sinking of the RMS Titanic in 1912 is one of history's most infamous disasters, and the dataset provides demographic and travel information about passengers aboard. The core question is:

> **Given a passenger's characteristics (age, class, sex, fare, etc.), can we predict whether they survived?**

Before any model can be trained, the raw data must be thoroughly cleaned, encoded, and scaled — which is the focus of this notebook.

---

## Dataset Details

| Property | Details |
|---|---|
| **Source** | `titanicDataset.zip` → `Titanic-Dataset.csv` |
| **Domain** | Historical passenger data, RMS Titanic (1912) |
| **Target Variable** | `Survived` (0 = No, 1 = Yes) |

### Features

| Column | Description | Type |
|---|---|---|
| `PassengerId` | Unique identifier for each passenger | Integer |
| `Survived` | Survival status (target) | Binary (0/1) |
| `Pclass` | Passenger ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Ordinal |
| `Name` | Full name of the passenger | String |
| `Sex` | Gender of the passenger | Categorical |
| `Age` | Age in years | Float |
| `SibSp` | Number of siblings/spouses aboard | Integer |
| `Parch` | Number of parents/children aboard | Integer |
| `Ticket` | Ticket number | String |
| `Fare` | Passenger fare | Float |
| `Cabin` | Cabin number | String |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Categorical |

### Missing Values (Pre-processing)

| Column | Missing % |
|---|---|
| `Cabin` | ~77% |
| `Age` | ~20% |
| `Embarked` | ~0.2% |

---

## Approach

The preprocessing pipeline follows these sequential steps:

### 1. Data Loading
The dataset is loaded from a ZIP archive and read into a Pandas DataFrame. Initial exploration covers shape, data types, and a sample of rows using `.head()` and `.info()`.

### 2. Missing Value Analysis
All columns are inspected for null values, with both absolute counts and percentages computed to guide imputation strategy.

### 3. Handling Missing Values
- **`Cabin`** — Dropped entirely due to ~77% missing data, making imputation unreliable.
- **`Age`** — Imputed with the **median age** to preserve the distribution and avoid skewing from outliers.
- **`Embarked`** — Imputed with the **mode** (most frequent port, Southampton) since only 2 rows were missing.

### 4. Encoding Categorical Features
- **`Sex`** — Label encoded as binary: `male → 0`, `female → 1`.
- **`Embarked`** — One-hot encoded into `Embarked_Q` and `Embarked_S` columns (`drop_first=True` avoids multicollinearity).

### 5. Dropping Irrelevant Features
Columns with no predictive value for machine learning were removed:
- `PassengerId` — Arbitrary identifier
- `Name` — Free text, not used in this pipeline
- `Ticket` — Alphanumeric, not directly useful
- `Sex` — Dropped at a later stage due to NaN propagation issues after mapping

### 6. Feature Scaling
`StandardScaler` from scikit-learn was applied to all numerical features to standardize them to zero mean and unit variance:
- `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`

This ensures no single feature dominates due to scale differences.

### 7. Removing Duplicates
Duplicate rows were identified and removed using `.drop_duplicates()` to ensure data integrity.

### 8. Data Type Conversion
One-hot encoded boolean columns (`Embarked_Q`, `Embarked_S`) were cast to `int` for compatibility with downstream ML models.

---

## Results

After the full preprocessing pipeline, the cleaned dataset has the following characteristics:

| Property | Value |
|---|---|
| **Missing Values** | 0 |
| **Duplicate Rows** | Removed |
| **Final Features** | `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked_Q`, `Embarked_S` |
| **Numerical Features Scaled** | Yes (StandardScaler) |
| **Categorical Features Encoded** | Yes (Label + One-Hot) |

The resulting DataFrame is clean, fully numerical, and ready to be fed into classification algorithms such as Logistic Regression, Decision Trees, or Random Forests for survival prediction.

---

## Project Structure

```
.
├── titanicDataset.zip        # Raw zipped dataset
├── Untitled17.ipynb          # Main preprocessing notebook
└── README.md                 # Project documentation
```

---

## Requirements

```
pandas
scikit-learn
```

Install via:

```bash
pip install pandas scikit-learn
```

---

## How to Run

1. Place `titanicDataset.zip` in `/content/` (or update the path in the notebook).
2. Open `Untitled17.ipynb` in Jupyter or Google Colab.
3. Run all cells sequentially.
