# Disease Prediction with ICU Mortality Prediction

## Project Overview
A Machine Learning-based ICU Mortality Prediction System that predicts whether a patient is likely to **Survive or Die** using 20 important ICU vitals and clinical parameters.

## Live Demo
🔗 [Open Streamlit App](https://akanksha-198-diseaseprediction-icu.streamlit.app/)

---

## Features
- ICU Mortality Prediction
- Survival Probability Analysis
- Interactive Streamlit UI
- Desktop GUI Application
- Multiple ML Model Comparison

---

## Models Compared

| Model | Accuracy | AUC-ROC |
|-------|-----------|----------|
| ✅ Random Forest | **80.21%** | **0.784** |
| XGBoost | 77.08% | 0.767 |
| SVM | 73.44% | 0.760 |
| Decision Tree | 68.75% | 0.620 |

---

## 20 Vitals Used
SOFA, Age, GCS, BUN, Creatinine, Urine Output, Lactate, pH, HCO3, MAP, HR, PaO2, FiO2, MechVent, WBC, HCT, Platelets, Albumin, Glucose, Weight

---

## Dataset
- Source: PhysioNet Challenge Dataset
- Total Patients: 958
- Train/Test Split: 80% / 20%
- Survivors: 737
- Non-Survivors: 221

---

## Tech Stack
Python • Scikit-learn • Streamlit • Tkinter • Pandas • NumPy

---

## Run Locally

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
