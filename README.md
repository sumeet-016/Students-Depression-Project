# Student Depression Predictor 🧠

**Machine Learning–based Early-Warning System for Student Mental Health**

This project builds a complete machine-learning pipeline and an interactive Streamlit application to identify students who may be at risk of depression.
It is designed strictly for **research, awareness, and support outreach**, not for medical diagnosis.

---

## 📁 Project Structure

| File / Folder                          | Description                                                        |
| -------------------------------------- | ------------------------------------------------------------------ |
| `Student Depression Dataset.csv`       | Main dataset used for training (27k+ rows).                        |
| `StudentDepression_update_dataset.csv` | Updated version (if used).                                         |
| `EDA.ipynb`                            | Exploratory analysis (visuals, distributions, correlations).       |
| `Model Training.ipynb`                 | Full ML pipeline, preprocessing, tuning, evaluation, model export. |
| `ada_boost_tuned_model.joblib`         | Saved ML pipeline (preprocessing + AdaBoost model).                |
| `app.py`                               | Streamlit web app for interactive single-entry prediction.         |
| `Mental Health.jpg`                    | Banner image for the web UI.                                       |
| `requirements.txt`                     | Needed Python dependencies.                                        |

---

## 🔍 Project Summary

* **Dataset size:** ~27,901 records
* **Columns:** 18 input features + 1 target (`Depression`)
* **Target:**

  * `1` → Depressed
  * `0` → Not depressed
* **Goal:** Early identification based on lifestyle, academic pressure, satisfaction, stress levels, and personal history.

---

## 📊 Dataset Features

Key features include:

* Demographics: Gender, Age, City, Degree
* Academic: CGPA, Academic Pressure, Study Satisfaction
* Work factors: Work Pressure, Work/Study Hours, Job Satisfaction
* Lifestyle: Sleep Duration, Dietary Habits
* Personal History: Suicidal thoughts, Family history of mental illness
* Target: **Depression**

Class distribution (slightly imbalanced):

* Depressed (`1`): **16,336**
* Not depressed (`0`): **11,565**

---

## 🧪 EDA Overview

`EDA.ipynb` includes:

* Distribution plots for numeric variables
* Countplots for categorical variables
* Correlation heatmaps
* Class imbalance analysis
* Outlier and missing value checks

---

## 🤖 Model Training

### Algorithms evaluated

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* AdaBoost

### Best Model

The best tuned model was **AdaBoost**, exported as:

```
ada_boost_tuned_model.joblib
```

### Tuned Hyperparameters

```python
n_estimators = 300
learning_rate = 1.0
algorithm = 'SAMME.R'
random_state = 42
```

A preprocessing pipeline (ColumnTransformer + OneHotEncoder + Scaler) is embedded inside the saved model.

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate:

* **Windows**

  ```bash
  venv\Scripts\activate
  ```
* **Mac/Linux**

  ```bash
  source venv/bin/activate
  ```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you face version issues, pin:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
streamlit
pillow
```

---

### 3️⃣ Run Jupyter notebooks

```bash
jupyter notebook
```

Execute:

1. `EDA.ipynb`
2. `Model Training.ipynb`

---

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

Your browser will open the interface.

---

## 🖼️ Streamlit App Features

The app (`app.py`):

* Loads the saved model safely with exception handling
* Displays the mental-health banner image
* Provides dropdowns/sliders for all input features
* Generates prediction + probability
* Alerts user if model file is missing
* Handles incorrect input dimensions gracefully

---

## 🧠 Predict Using Python Script

```python
import joblib
import pandas as pd

model = joblib.load('ada_boost_tuned_model.joblib')

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
proba = model.predict_proba(sample)[:, 1]

print("Prediction:", pred[0])
print("Depression Probability:", proba[0])
```

---

## 🛠️ Troubleshooting

### ❗Model not loading?

Your app will show a clean error, but check console logs for details.

Common cause → **scikit-learn version mismatch**

Check version:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Install matching version (example):

```bash
pip install scikit-learn==1.2.2
```

---

## 🧭 Ethical Considerations

* **This is NOT a diagnostic tool.**
* Use results to **identify students who may need support or outreach**.
* Do NOT use for academic penalties or judgement.
* Ensure **data privacy, consent, confidentiality**.
* Evaluate for **bias or fairness issues** before real deployment.

---

## 🚀 Future Improvements

* Add SHAP/LIME explainability
* Add a Flask/Streamlit dashboard for batch predictions
* Add Docker container for consistent deployment
* Add unit tests + CI automation
* Include advanced fairness metrics (group-wise F1, DP, EO)

---

## 👤 Author

**Sumeet Kumar Pal**

* GitHub: **sumeet-016**
* LinkedIn: [https://www.linkedin.com/in/palsumeet/](https://www.linkedin.com/in/palsumeet/)

---
