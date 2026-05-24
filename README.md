# 🏥 ICU Mortality Predictor

## Project Overview
Machine Learning based ICU patient mortality prediction system.  
Predicts whether a patient will **Survive or Die** based on 20 critical ICU vitals.

## Live Demo
🔗 [Click here to open the app](https://your-app.streamlit.app) ← *deploy ke baad link update karo*

## Team Members
- Member 1
- Member 2
- Member 3

## Models Compared
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| ✅ **Random Forest** | **80.21%** | **0.784** |
| XGBoost | 77.08% | 0.767 |
| SVM | 73.44% | 0.760 |
| Decision Tree | 68.75% | 0.620 |

## 20 Vitals Used for Prediction
| # | Vital | Category |
|---|-------|----------|
| 1 | SOFA | Severity Score |
| 2 | Age | Demographics |
| 3 | GCS | Neurological |
| 4 | BUN | Kidney |
| 5 | Creatinine | Kidney |
| 6 | Urine Output | Kidney |
| 7 | Lactate | Metabolic |
| 8 | pH | Acid-Base |
| 9 | HCO3 | Acid-Base |
| 10 | MAP | Hemodynamics |
| 11 | HR | Hemodynamics |
| 12 | PaO2 | Respiratory |
| 13 | FiO2 | Respiratory |
| 14 | MechVent | Respiratory |
| 15 | WBC | Blood |
| 16 | HCT | Blood |
| 17 | Platelets | Blood |
| 18 | Albumin | Liver/Protein |
| 19 | Glucose | Metabolic |
| 20 | Weight | Demographics |

## Dataset
- **Source:** PhysioNet Challenge
- **Total patients:** 958
- **Train:** 766 patients (80%)
- **Test:** 192 patients (20%)
- **Class split:** 737 Survived / 221 Died

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Run Desktop App
```bash
pip install scikit-learn pandas
python full.py
```

## Tech Stack
Python | Scikit-learn | Random Forest | Streamlit | Tkinter | Pandas | NumPy

## Project Structure
```
ICU-Mortality-Predictor/
├── app.py              ← Streamlit web app
├── full.py             ← Tkinter desktop app
├── merged_output.csv   ← Dataset
├── requirements.txt    ← Dependencies
└── README.md           ← This file
```
