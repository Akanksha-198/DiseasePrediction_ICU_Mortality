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

# ── EXACT SAME COLORS AS full.py ──────
# BG      = "#F7F9FC"
# CARD    = "#FFFFFF"
# PRIMARY = "#A8DADC"
# ACCENT  = "#FFC8DD"
# TEXT    = "#2b2d42"

st.markdown("""
<style>
    /* ── Background same as full.py BG="#F7F9FC" ── */
    .stApp {
        background-color: #F7F9FC;
    }

    /* ── Main content card area WHITE ── */
    .block-container {
        background-color: #F7F9FC;
        padding-top: 0rem;
    }

    /* ── Title bar — PRIMARY = #A8DADC ── */
    .title-bar {
        background-color: #A8DADC;
        padding: 18px 24px;
        margin: -1rem -1rem 1.5rem -1rem;
        text-align: center;
    }
    .title-bar h1 {
        font-family: "Segoe UI", sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #2b2d42;
        margin: 0;
    }

    /* ── Accuracy banner — same teal strip ── */
    .acc-banner {
        background-color: #d0f0f0;
        padding: 6px 16px;
        color: #1a4a4a;
        font-size: 13px;
        font-family: "Segoe UI", sans-serif;
        text-align: center;
        margin-bottom: 1rem;
        border-radius: 4px;
    }

    /* ── Section headers — same PRIMARY ── */
    .section-hdr {
        background-color: #A8DADC;
        padding: 7px 14px;
        border-radius: 4px;
        font-family: "Segoe UI", sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: #2b2d42;
        margin: 14px 0 8px 0;
    }

    /* ── Input fields — same as entry bg "#eef2f3" ── */
    input[type="number"], .stSelectbox select {
        background-color: #eef2f3 !important;
        font-family: "Segoe UI", sans-serif !important;
    }
    .stNumberInput input {
        background-color: #eef2f3 !important;
    }

    /* ── Labels bold Segoe UI same as full.py ── */
    label, .stNumberInput label, .stSelectbox label {
        font-family: "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        color: #2b2d42 !important;
        font-size: 13px !important;
    }

    /* ── Predict button — PRIMARY color ── */
    .stButton > button[kind="primary"] {
        background-color: #A8DADC !important;
        color: #2b2d42 !important;
        font-family: "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #88c8ca !important;
    }

    /* ── Refresh / secondary button — ACCENT = #FFC8DD ── */
    .stButton > button[kind="secondary"] {
        background-color: #FFC8DD !important;
        color: #2b2d42 !important;
        font-family: "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 4px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #ffaacb !important;
    }

    /* ── Result boxes ── */
    .result-dead {
        background: #ffe0e0;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-family: "Segoe UI", sans-serif;
    }
    .result-survive {
        background: #e0ffe0;
        border: 2px solid #44bb44;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-family: "Segoe UI", sans-serif;
    }

    /* ── Conditions box — same "#eaf4f4" ── */
    .cond-box {
        background-color: #eaf4f4;
        border-radius: 6px;
        padding: 14px 18px;
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        color: #2b2d42;
        min-height: 90px;
    }

    /* ── Risk scale bar ── */
    .risk-bar-wrap {
        margin: 10px 0 4px 0;
    }
    .risk-bar {
        width: 100%;
        height: 28px;
        border-radius: 5px;
        background: linear-gradient(to right, #00ff00, #ffff00, #ff0000);
        position: relative;
    }
    .risk-needle {
        position: absolute;
        top: 2px;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #2b2d42;
        border: 2px solid white;
        transform: translateX(-50%);
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }
    .risk-labels {
        display: flex;
        justify-content: space-between;
        font-family: "Segoe UI", sans-serif;
        font-size: 12px;
        margin-top: 3px;
    }
    .risk-labels .low  { color: green; }
    .risk-labels .med  { color: orange; }
    .risk-labels .high { color: red; }

    /* hide streamlit default header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── FEATURES ──────────────────────────
FEATURES = [
    'SOFA','Age','GCS','BUN','Creatinine','Urine','Lactate',
    'pH','HCO3','MAP','HR','PaO2','FiO2','MechVent',
    'WBC','HCT','Platelets','Albumin','Glucose','Weight'
]
TARGET = 'In-hospital_death'

# ── LOAD + TRAIN (cached) ─────────────
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
    X = df[FEATURES]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred) * 100
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return model, acc, auc, len(X_train), len(X_test), tn, fp, fn, tp

model, acc, auc, n_train, n_test, tn, fp, fn, tp = load_and_train()

# ── TITLE BAR (same as full.py PRIMARY header) ──
st.markdown("""
<div class="title-bar">
  <h1>ICU Mortality Predictor</h1>
</div>
""", unsafe_allow_html=True)

# ── ACCURACY BANNER ───────────────────
st.markdown(f"""
<div class="acc-banner">
  Model: Random Forest &nbsp;|&nbsp;
  Train: {n_train} patients &nbsp;|&nbsp;
  Test: {n_test} patients &nbsp;|&nbsp;
  Accuracy: {acc:.1f}% &nbsp;|&nbsp;
  AUC: {auc:.3f}
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────
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
    st.markdown("### ℹ️ Dataset")
    st.info(f"Total: 958 patients\nTrain: {n_train} (80%)\nTest: {n_test} (20%)\nFeatures: 20 vitals")

# ══════════════════════════════════════
# INPUT GRID  (same layout as full.py)
# ══════════════════════════════════════
st.markdown('<div class="section-hdr">🔴 Severity & Neurological</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: SOFA = st.number_input("SOFA Score (0–24)", 0, 24, 9, 1)
with c2: Age  = st.number_input("Age (years)",       18, 100, 63)
with c3: GCS  = st.number_input("GCS (3–15)",         3,  15, 10, 1)

st.markdown('<div class="section-hdr">🫘 Kidney Function</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: BUN        = st.number_input("BUN (mg/dL)",          1.0, 150.0,  22.0, 0.1)
with c2: Creatinine = st.number_input("Creatinine (mg/dL)",   0.1,  15.0,   1.0, 0.01)
with c3: Urine      = st.number_input("Urine Output (mL/hr)", 0.0,3000.0,  95.0, 1.0)

st.markdown('<div class="section-hdr">⚗️ Metabolic & Acid-Base</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: Lactate = st.number_input("Lactate (mmol/L)", 0.5,  25.0,  2.0, 0.1)
with c2: pH      = st.number_input("pH (arterial)",    6.8,   7.6, 7.38, 0.01)
with c3: HCO3    = st.number_input("HCO3 (mEq/L)",    5.0,  50.0, 22.0, 0.1)
with c4: Glucose = st.number_input("Glucose (mg/dL)", 50.0, 400.0,135.0, 1.0)

st.markdown('<div class="section-hdr">🫀 Hemodynamics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: MAP = st.number_input("MAP (mmHg)",        40.0, 200.0, 79.0, 0.1)
with c2: HR  = st.number_input("Heart Rate (bpm)",    30,   180,   89,    1)

st.markdown('<div class="section-hdr">🫁 Respiratory</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: PaO2     = st.number_input("PaO2 (mmHg)",      40.0, 500.0, 132.0, 1.0)
with c2: FiO2     = st.number_input("FiO2 (0.21–1.0)",  0.21,   1.0,  0.54, 0.01)
with c3: MechVent = st.selectbox("Mechanical Ventilation", [1, 0],
                                  format_func=lambda x: "Yes (ON)" if x==1 else "No (OFF)")

st.markdown('<div class="section-hdr">🩸 Blood & Lab Values</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: WBC       = st.number_input("WBC (k/μL)",      0.1, 150.0,  12.2, 0.1)
with c2: HCT       = st.number_input("HCT (%)",         15.0,  60.0,  31.0, 0.1)
with c3: Platelets = st.number_input("Platelets (k/μL)",10.0,1000.0, 181.0, 1.0)
with c4: Albumin   = st.number_input("Albumin (g/dL)",   1.0,   5.0,   2.8, 0.1)
with c5: Weight    = st.number_input("Weight (kg)",      30.0, 250.0,  82.0, 0.1)

# ══════════════════════════════════════
# BUTTONS — Predict + Refresh (same as full.py)
# ══════════════════════════════════════
st.markdown("---")
col_pred, col_ref = st.columns(2)
with col_pred:
    predict_clicked = st.button("Predict", use_container_width=True, type="primary")
with col_ref:
    refresh_clicked = st.button("Refresh", use_container_width=True, type="secondary")

if refresh_clicked:
    st.rerun()

# ══════════════════════════════════════
# RESULT SECTION
# ══════════════════════════════════════
if predict_clicked:
    input_data = [[SOFA, Age, GCS, BUN, Creatinine, Urine, Lactate,
                   pH, HCO3, MAP, HR, PaO2, FiO2, MechVent,
                   WBC, HCT, Platelets, Albumin, Glucose, Weight]]

    df_input = pd.DataFrame(input_data, columns=FEATURES)
    prob     = model.predict_proba(df_input)[0][1]
    pred     = 1 if prob > 0.4 else 0
    user     = dict(zip(FEATURES, input_data[0]))

    st.markdown("---")

    # ── Prediction label (same as status_label in full.py) ──
    result_text = "YES (Patient may DIE)" if pred == 1 else "NO (Patient will SURVIVE)"
    result_color = "red" if pred == 1 else "green"
    st.markdown(f"""
    <p style="text-align:center; font-family:'Segoe UI',sans-serif;
              font-size:22px; font-weight:700; color:{result_color}; margin:0;">
        Prediction: {result_text}
    </p>""", unsafe_allow_html=True)

    # ── Probability label ──
    st.markdown(f"""
    <p style="text-align:center; font-family:'Segoe UI',sans-serif;
              font-size:15px; color:#2b2d42; margin:4px 0 12px 0;">
        Probability: {prob*100:.1f}%
    </p>""", unsafe_allow_html=True)

    # ── Risk scale bar (green→red, same as canvas in full.py) ──
    needle_pct = min(max(prob * 100, 1), 99)
    st.markdown(f"""
    <div class="risk-bar-wrap">
      <div class="risk-bar">
        <div class="risk-needle" style="left:{needle_pct}%;"></div>
      </div>
      <div class="risk-labels">
        <span class="low">Low Risk</span>
        <span class="med">Medium</span>
        <span class="high">High Risk</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Conditions box (same "#eaf4f4" background) ──
    diseases = []
    if user['Lactate'] > 2 and user['MAP'] < 65:
        diseases.append("Shock")
    if user['Creatinine'] > 1.5 and user['BUN'] > 25:
        diseases.append("Kidney Failure")
    if user['GCS'] < 8:
        diseases.append("Neurological Issue")
    if user['PaO2'] < 60 or user['FiO2'] > 0.6:
        diseases.append("Respiratory Failure")
    if user['WBC'] > 12:
        diseases.append("Infection / Sepsis")
    if user['SOFA'] >= 11:
        diseases.append("Multi-Organ Failure")
    if user['pH'] < 7.35:
        diseases.append("Metabolic Acidosis")

    cond_text = "\n".join(f"• {d}" for d in diseases) if diseases else "• No major issues"
    st.markdown(f"""
    <div class="cond-box">
        <b>Conditions:</b><br>
        <pre style="margin:6px 0 0 0; font-family:'Segoe UI',sans-serif;
                    font-size:13px; background:transparent; border:none;">{cond_text}</pre>
    </div>
    """, unsafe_allow_html=True)

    # ── Vital flags expander ──
    with st.expander("📋 Vital Sign Flags — click to expand"):
        normals = {
            'SOFA':(0,6),'GCS':(13,15),'BUN':(7,20),
            'Creatinine':(0.6,1.2),'Urine':(30,400),
            'Lactate':(0.5,2.0),'pH':(7.35,7.45),'HCO3':(22,28),
            'MAP':(70,100),'HR':(60,100),'PaO2':(80,300),
            'FiO2':(0.21,0.40),'WBC':(4,11),'HCT':(36,50),
            'Platelets':(150,400),'Albumin':(3.5,5.0),'Glucose':(70,140),
        }
        for feat, val in zip(FEATURES, input_data[0]):
            if feat in ('MechVent','Age','Weight'): continue
            lo, hi = normals.get(feat, (0, 9999))
            if val > hi:
                st.error(f"⬆️ {feat}: {val:.2f} — HIGH  (normal: {lo}–{hi})")
            elif val < lo:
                st.info(f"⬇️ {feat}: {val:.2f} — LOW   (normal: {lo}–{hi})")
            else:
                st.success(f"✅ {feat}: {val:.2f} — Normal")

st.markdown("---")
st.caption("⚠️ For educational/research use only. Not a substitute for clinical judgement. "
           "| Random Forest · 20 ICU Vitals · 958 patients")
