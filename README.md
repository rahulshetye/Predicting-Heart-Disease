
#  Predicting Heart Disease – Kaggle Playground Series (S6E2)

This project is a machine learning solution for the Kaggle competition:

**Predicting Heart Disease – Playground Series Season 6 Episode 2**

The objective is to predict the probability of heart disease using tabular medical data.
The evaluation metric used in the competition is **ROC-AUC Score**.

---

##  Problem Statement

Given patient medical attributes such as:

* Age
* Sex
* Chest Pain Type
* Blood Pressure
* Cholesterol
* Max Heart Rate
* ST Depression
* Number of Vessels
* Thallium Test Result
* and others

The task is to predict the likelihood of **Heart Disease (Presence or Absence)**.

Submissions must provide probabilities for the positive class.

---

##  Approach

### 1️⃣ Data Preprocessing

* Loaded CSV dataset using pandas
* Encoded target variable:

  * `Presence → 1`
  * `Absence → 0`
* Separated features (X) and target (y)

---

### 2️⃣ Model Used

**LightGBM (LGBMClassifier)**

LightGBM is a Gradient Boosting framework that:

* Builds multiple decision trees
* Learns from previous errors
* Optimizes prediction performance
* Works extremely well for tabular datasets

---

### 3️⃣ Validation Strategy

Used **Stratified 5-Fold Cross Validation**

Why stratified?

* Ensures class balance in each fold
* Prevents biased validation results

Why cross-validation?

* Reduces overfitting
* Provides robust performance estimation
* Simulates unseen data prediction

---

### 4️⃣ Early Stopping

Used early stopping during training:

* Stops training if validation AUC does not improve
* Prevents overfitting
* Improves generalization

---

##  Evaluation Metric

The competition uses:

> **ROC-AUC (Receiver Operating Characteristic – Area Under Curve)**

Why ROC-AUC?

* Measures ranking ability of the model
* Works well for imbalanced datasets
* Evaluates probability predictions

---

##  Project Structure

```
├── train.csv
├── test.csv
├── notebook.ipynb
├── submission.csv
└── README.md
```

---

##  How To Run (Google Colab)

1. Install dependencies:

```
pip install lightgbm
```

2. Upload dataset files

3. Run training notebook

4. Generate submission file:

```
submission.csv
```

5. Upload to Kaggle

---

##  Model Parameters

```python
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

##  Skills Demonstrated

* Data preprocessing
* Feature-target separation
* Binary classification
* Cross-validation
* Gradient boosting
* Model evaluation (ROC-AUC)
* Kaggle submission formatting
* Handling feature mismatch errors

---

##  Possible Improvements

* Hyperparameter tuning (Optuna)
* Feature engineering
* Ensembling (LightGBM + XGBoost)
* CatBoost experimentation
* Feature importance analysis
* Stacking models

---

##  Author

Rahul Shetye

Built as part of Kaggle Playground competition to strengthen machine learning and tabular modeling skills.

---

##  Citation

Yao Yan, Walter Reade, Elizabeth Park.
Predicting Heart Disease. Kaggle, 2026.

---

## Link of Code
https://colab.research.google.com/drive/1LbSPgtNsKmfSS_sHi2EvSB5WdeSqcCHk?usp=sharing

---

## Link of Compition
https://www.kaggle.com/competitions/playground-series-s6e2/overview


