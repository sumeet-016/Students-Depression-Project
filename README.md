# Student Depression Prediction Using Machine Learning

## Overview

This repository implements a machine‑learning pipeline to identify students at risk of depression using academic, behavioral, demographic, and lifestyle features. The goal is to provide an early‑warning analytic (not a clinical diagnosis) that institutions or counselors can use to prioritize outreach and support.

---

## Quick project summary

* **Dataset**: `Student Depression Dataset.csv` (also an updated version `StudentDepression_update_dataset.csv`).
* **Records**: 27,901 rows, 18 columns.
* **Target column**: `Depression` (binary: `1` = depressed, `0` = not depressed).
* **Saved model**: `ada_boost_tuned_model.joblib` (pipeline containing preprocessing + AdaBoost classifier).
* **Main notebooks**: `EDA.ipynb`, `Model Training.ipynb`.

---

## Dataset (columns)

The dataset contains the following columns:

* id
* Gender
* Age
* City
* Profession
* Academic Pressure
* Work Pressure
* CGPA
* Study Satisfaction
* Job Satisfaction
* Sleep Duration
* Dietary Habits
* Degree
* Have you ever had suicidal thoughts ?
* Work/Study Hours
* Financial Stress
* Family History of Mental Illness
* Depression (target)

**Target distribution (original dataset)**: `Depression` = 1: **16,336** rows, 0: **11,565** rows — slightly imbalanced toward the positive class.

---

## Preprocessing and EDA

* Performed standard exploratory analysis (distribution plots, correlation heatmaps, class balance checks) in `EDA.ipynb`.
* Preprocessing pipeline (used in `Model Training.ipynb`) handles:

  * Encoding categorical variables (one‑hot / ordinal as appropriate)
  * Scaling numerical features
  * Missing value handling (if any)
  * ColumnTransformer + Pipeline to bundle preprocessing with the estimator

---

## Models evaluated

The notebooks train and evaluate multiple baseline and tuned models (examples):

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* AdaBoost

### Tuned models (found in `Model Training.ipynb`)

**Gradient Boosting (tuned)**

* `subsample=0.8`
* `n_estimators=250`
* `min_samples_split=20`
* `min_samples_leaf=8`
* `max_features='sqrt'`
* `max_depth=2`
* `learning_rate=0.1`

**AdaBoost (tuned and saved)**

* `n_estimators=300`
* `learning_rate=1.0`
* `algorithm='SAMME.R'`
* `random_state=42`

The AdaBoost pipeline was exported as `ada_boost_tuned_model.joblib`.

---

## How to run (recommended)

1. Create a Python environment and install dependencies. A `requirements.txt` is recommended (if not present, create one with the packages below):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# If requirements.txt is missing, at minimum install:
# pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

2. Open and run the notebooks in order:

* `EDA.ipynb` — exploratory analysis and visualizations
* `Model Training.ipynb` — preprocessing, model training, evaluation, and model export

3. Make predictions with the saved pipeline (example):

```python
import joblib
import pandas as pd

model = joblib.load('ada_boost_tuned_model.joblib')
# Prepare a single-row DataFrame with the same feature columns (drop `id` and `Depression`)
sample = pd.DataFrame([{
    'Gender':'Female',
    'Age':21,
    'City':'Mumbai',
    'Profession':'Student',
    'Academic Pressure':3,
    'Work Pressure':2,
    'CGPA':7.5,
    'Study Satisfaction':4,
    'Job Satisfaction':0,
    'Sleep Duration':'5-6 hours',
    'Dietary Habits':'Moderate',
    'Degree':'BSc',
    'Have you ever had suicidal thoughts ?':'No',
    'Work/Study Hours':6,
    'Financial Stress':2,
    'Family History of Mental Illness':'No'
}])

pred = model.predict(sample)
print('Predicted depression label:', pred[0])
```

**Note on sklearn/joblib compatibility:** If `joblib.load` raises an error about missing attributes or unconventional objects (for example when a saved pipeline uses a different scikit‑learn version than your runtime), run the following inside the same environment where the model was trained (or install a compatible `scikit-learn` version):

```bash
python -c "import sklearn; print(sklearn.__version__)"
# then install the matching version, for example:
# pip install scikit-learn==1.2.2
```

---

## Reproducibility & evaluation

* Metrics printed in `Model Training.ipynb` include accuracy, precision, recall, F1 and the confusion matrix for each trained model. Use the notebooks to reproduce metric tables and plots.
* Random seeds (`random_state=42`) were used in tuned classifiers for reproducibility.

---

## Ethical considerations

* This tool **is not a clinical diagnostic** — it is intended for prioritizing outreach and research.
* It must not be used for punitive action (discipline, academic penalties) without human verification.
* Consider privacy, consent, and data protection before deployment.

---

## Next steps / improvements

* Add a `requirements.txt` or `environment.yml` to lock dependencies.
* Create a small Flask/Streamlit app for inference and accessible dashboards.
* Add SHAP or LIME interpretability reports to explain predictions.
* Add automated unit tests and CI for model reproducibility.
* Run a bias and fairness audit (group-wise metrics, demographic parity checks).

---

## Contact

If you want this README written back to `README.md` in the repo or want the short `README.md` cleaned up directly, tell me and I will update the file.

*I reviewed your current README as a baseline.* fileciteturn3file0
