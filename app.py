import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="ICU Mortality Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #A8DADC, #457B9D);
        padding: 20px; border-radius: 10px;
        text-align: center; color: white; margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f8ff; border-radius: 8px;
        padding: 15px; text-align: center;
        border: 1px solid #A8DADC;
    }
    .result-dead {
        background: #ffe0e0; border: 2px solid #ff4444;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .result-survive {
        background: #e0ffe0; border: 2px solid #44bb44;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .section-header {
        background: #A8DADC; padding: 8px 15px;
        border-radius: 5px; font-weight: bold;
        color: #2b2d42; margin: 15px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── FEATURES ──────────────────────────
FEATURES = [
    'SOFA','Age','GCS','BUN','Creatinine','Urine','Lactate',
    'pH','HCO3','MAP','HR','PaO2','FiO2','MechVent',
    'WBC','HCT','Platelets','Albumin','Glucose','Weight'
]
TARGET = 'In-hospital_death'

# ── LOAD + TRAIN ──────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("merged_output.csv")
    df.drop(columns=['TropI', 'TropT'], inplace=True)
    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.85]
    df.drop(columns=high_null, inplace=True)
    drop_cols = ['RecordID', 'Survival', 'Length_of_stay', 'completeness']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(0, inplace=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred) * 100
    auc    = roc_auc_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return model, acc, auc, len(X_train), len(X_test), tn, fp, fn, tp

model, acc, auc, n_train, n_test, tn, fp, fn, tp = load_and_train()

# ── HEADER ────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 ICU Mortality Predictor</h1>
    <p>Machine Learning based ICU Patient Survival Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── MODEL STATS ───────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", "Random Forest")
with col2:
    st.metric("Accuracy", f"{acc:.1f}%")
with col3:
    st.metric("AUC-ROC", f"{auc:.3f}")
with col4:
    st.metric("Train / Test", f"{n_train} / {n_test}")

st.markdown("---")

# ── SIDEBAR INFO ──────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Accuracy | **{acc:.2f}%** |
    | AUC-ROC | **{auc:.4f}** |
    | Sensitivity | **{tp/(tp+fn)*100:.1f}%** |
    | Specificity | **{tn/(tn+fp)*100:.1f}%** |
    """)
    st.markdown("### 📋 Confusion Matrix")
    st.markdown(f"""
    | | Pred Survived | Pred Died |
    |--|--|--|
    | **Act. Survived** | {tn} ✅ | {fp} ❌ |
    | **Act. Died** | {fn} ❌ | {tp} ✅ |
    """)
    st.markdown("### ℹ️ Dataset Info")
    st.info(f"Total: 958 patients\nTrain: {n_train} (80%)\nTest: {n_test} (20%)\nFeatures: 20 vitals")

# ── INPUT FIELDS ──────────────────────
st.markdown("### 📝 Enter Patient Vitals")

# Row 1 — Severity & Neuro
st.markdown('<div class="section-header">🔴 Severity & Neurological</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    SOFA = st.number_input("SOFA Score (0-24)", min_value=0, max_value=24, value=9, step=1,
                            help="Sequential Organ Failure Assessment")
with c2:
    Age  = st.number_input("Age (years)", min_value=18, max_value=100, value=63)
with c3:
    GCS  = st.number_input("GCS (3-15)", min_value=3, max_value=15, value=10, step=1,
                            help="Glasgow Coma Scale — 15=alert, 3=coma")

# Row 2 — Kidney
st.markdown('<div class="section-header">🫘 Kidney Function</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    BUN        = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=150.0, value=22.0, step=0.1)
with c2:
    Creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.01)
with c3:
    Urine      = st.number_input("Urine Output (mL/hr)", min_value=0.0, max_value=3000.0, value=95.0, step=1.0)

# Row 3 — Metabolic
st.markdown('<div class="section-header">⚗️ Metabolic & Acid-Base</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    Lactate = st.number_input("Lactate (mmol/L)", min_value=0.5, max_value=25.0, value=2.0, step=0.1)
with c2:
    pH      = st.number_input("pH (arterial)", min_value=6.8, max_value=7.6, value=7.38, step=0.01)
with c3:
    HCO3    = st.number_input("HCO3 (mEq/L)", min_value=5.0, max_value=50.0, value=22.0, step=0.1)
with c4:
    Glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=400.0, value=135.0, step=1.0)

# Row 4 — Hemodynamics
st.markdown('<div class="section-header">🫀 Hemodynamics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    MAP = st.number_input("MAP (mmHg)", min_value=40.0, max_value=200.0, value=79.0, step=0.1)
with c2:
    HR  = st.number_input("Heart Rate (bpm)", min_value=30, max_value=180, value=89, step=1)

# Row 5 — Respiratory
st.markdown('<div class="section-header">🫁 Respiratory</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    PaO2    = st.number_input("PaO2 (mmHg)", min_value=40.0, max_value=500.0, value=132.0, step=1.0)
with c2:
    FiO2    = st.number_input("FiO2 (0.21-1.0)", min_value=0.21, max_value=1.0, value=0.54, step=0.01)
with c3:
    MechVent = st.selectbox("Mechanical Ventilation", options=[1, 0],
                             format_func=lambda x: "Yes (ON)" if x == 1 else "No (OFF)")

# Row 6 — Blood
st.markdown('<div class="section-header">🩸 Blood & Lab Values</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    WBC      = st.number_input("WBC (k/μL)", min_value=0.1, max_value=150.0, value=12.2, step=0.1)
with c2:
    HCT      = st.number_input("HCT (%)", min_value=15.0, max_value=60.0, value=31.0, step=0.1)
with c3:
    Platelets = st.number_input("Platelets (k/μL)", min_value=10.0, max_value=1000.0, value=181.0, step=1.0)
with c4:
    Albumin  = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=5.0, value=2.8, step=0.1)
with c5:
    Weight   = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=82.0, step=0.1)

# ── PREDICT BUTTON ────────────────────
st.markdown("---")
if st.button("🔍 Predict Mortality", use_container_width=True, type="primary"):

    input_data = [[SOFA, Age, GCS, BUN, Creatinine, Urine, Lactate,
                   pH, HCO3, MAP, HR, PaO2, FiO2, MechVent,
                   WBC, HCT, Platelets, Albumin, Glucose, Weight]]

    df_input   = pd.DataFrame(input_data, columns=FEATURES)
    prob       = model.predict_proba(df_input)[0][1]
    pred       = 1 if prob > 0.4 else 0

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col_res, col_prob = st.columns([2, 1])

    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div class="result-dead">
                <h2>⚠️ HIGH MORTALITY RISK</h2>
                <h3>Prediction: Patient may DIE</h3>
                <p>Death Probability: <b>{prob*100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-survive">
                <h2>✅ LOWER MORTALITY RISK</h2>
                <h3>Prediction: Patient will SURVIVE</h3>
                <p>Survival Probability: <b>{(1-prob)*100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

    with col_prob:
        st.metric("Death Probability",   f"{prob*100:.1f}%")
        st.metric("Survival Probability", f"{(1-prob)*100:.1f}%")
        risk = "🔴 HIGH" if prob > 0.6 else "🟡 MODERATE" if prob > 0.35 else "🟢 LOW"
        st.metric("Risk Level", risk)

    # ── DISEASE CONDITIONS ────────────
    st.markdown("### 🏥 Detected Conditions")
    diseases = []
    user = dict(zip(FEATURES, input_data[0]))

    if user['Lactate'] > 2 and user['MAP'] < 65:
        diseases.append(("🔴 Shock", "Lactate high + MAP low — hemodynamic instability"))
    if user['Creatinine'] > 1.5 and user['BUN'] > 25:
        diseases.append(("🟠 Kidney Failure", "Creatinine & BUN elevated — acute kidney injury"))
    if user['GCS'] < 8:
        diseases.append(("🟣 Neurological Issue", "GCS < 8 — severe brain dysfunction / coma"))
    if user['PaO2'] < 60 or user['FiO2'] > 0.6:
        diseases.append(("🔵 Respiratory Failure", "Low PaO2 or high FiO2 need — lung failure"))
    if user['WBC'] > 12:
        diseases.append(("🟡 Infection / Sepsis", "WBC elevated — possible infection"))
    if user['SOFA'] >= 11:
        diseases.append(("🔴 Multi-Organ Failure", "SOFA ≥ 11 — severe multi-organ dysfunction"))
    if user['pH'] < 7.35:
        diseases.append(("🟠 Metabolic Acidosis", "pH < 7.35 — acid-base imbalance"))

    if diseases:
        for name, desc in diseases:
            st.warning(f"**{name}** — {desc}")
    else:
        st.success("✅ No major conditions detected")

    # ── VITAL FLAGS ───────────────────
    with st.expander("📋 Vital Sign Flags"):
        normals = {
            'SOFA':(0,6),'Age':(0,200),'GCS':(13,15),
            'BUN':(7,20),'Creatinine':(0.6,1.2),'Urine':(30,400),
            'Lactate':(0.5,2.0),'pH':(7.35,7.45),'HCO3':(22,28),
            'MAP':(70,100),'HR':(60,100),'PaO2':(80,300),
            'FiO2':(0.21,0.40),'WBC':(4,11),'HCT':(36,50),
            'Platelets':(150,400),'Albumin':(3.5,5.0),'Glucose':(70,140),
            'Weight':(0,300)
        }
        for feat, val in zip(FEATURES, input_data[0]):
            if feat == 'MechVent': continue
            lo, hi = normals.get(feat, (0, 9999))
            if val > hi:
                st.error(f"⬆️ {feat}: {val:.2f} — HIGH (normal: {lo}–{hi})")
            elif val < lo:
                st.info(f"⬇️ {feat}: {val:.2f} — LOW (normal: {lo}–{hi})")
            else:
                st.success(f"✅ {feat}: {val:.2f} — Normal")

st.markdown("---")
st.caption("⚠️ For educational/research use only. Not a substitute for clinical judgement. | Random Forest · 20 ICU Vitals · 958 patients")
