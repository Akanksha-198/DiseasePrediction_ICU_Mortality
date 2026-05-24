import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ICU Mortality Predictor",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    /* ── Global background ── */
    .stApp { background-color: #F7F9FC; }
    .block-container {
        background-color: #F7F9FC;
        padding-top: 0 !important;
        max-width: 820px;
    }

    /* ── Hide streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Title bar exactly like full.py ── */
    .title-bar {
        background-color: #A8DADC;
        padding: 14px 0;
        text-align: center;
        margin: -4rem -4rem 1rem -4rem;
    }
    .title-bar h2 {
        font-family: "Segoe UI", sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #2b2d42;
        margin: 0;
    }

    /* ── Card (white box) ── */
    .form-card {
        background: #FFFFFF;
        border-radius: 4px;
        padding: 18px 24px;
        margin-bottom: 14px;
    }

    /* ── Input row — label + box side by side ── */
    .input-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .input-label {
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: #2b2d42;
        width: 110px;
        flex-shrink: 0;
    }

    /* ── Streamlit number input styling ── */
    .stNumberInput input {
        background-color: #eef2f3 !important;
        font-family: "Segoe UI", sans-serif !important;
        font-size: 13px !important;
        color: #2b2d42 !important;
        border: 1px solid #ccc !important;
        border-radius: 3px !important;
        padding: 4px 8px !important;
    }
    .stSelectbox > div > div {
        background-color: #eef2f3 !important;
        font-family: "Segoe UI", sans-serif !important;
    }

    /* ── Hide number input label (we use custom) ── */
    .stNumberInput label { display: none !important; }
    .stSelectbox label   { display: none !important; }

    /* ── Predict button ── */
    div[data-testid="column"]:nth-child(1) .stButton > button {
        background-color: #A8DADC !important;
        color: #2b2d42 !important;
        font-family: "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 3px !important;
        width: 100% !important;
        padding: 8px !important;
    }
    div[data-testid="column"]:nth-child(1) .stButton > button:hover {
        background-color: #88c8ca !important;
    }

    /* ── Refresh button ── */
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background-color: #FFC8DD !important;
        color: #2b2d42 !important;
        font-family: "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 3px !important;
        width: 100% !important;
        padding: 8px !important;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button:hover {
        background-color: #ffaacb !important;
    }

    /* ── Result text ── */
    .pred-text {
        font-family: "Segoe UI", sans-serif;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin: 8px 0 2px 0;
    }
    .prob-text {
        font-family: "Segoe UI", sans-serif;
        font-size: 14px;
        text-align: center;
        color: #2b2d42;
        margin-bottom: 8px;
    }

    /* ── Risk bar ── */
    .risk-bar {
        width: 100%;
        height: 36px;
        border-radius: 4px;
        background: linear-gradient(to right, #00cc00, #ffff00, #ff0000);
        position: relative;
        margin: 6px 0 2px 0;
    }
    .risk-needle {
        position: absolute;
        top: 4px;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #1a1a2e;
        border: 2px solid white;
        transform: translateX(-50%);
        box-shadow: 0 1px 5px rgba(0,0,0,0.5);
    }
    .risk-labels {
        display: flex;
        justify-content: space-between;
        font-family: "Segoe UI", sans-serif;
        font-size: 12px;
        margin-top: 2px;
        padding: 0 2px;
    }

    /* ── Conditions box ── */
    .cond-box {
        background-color: #eaf4f4;
        border-radius: 4px;
        padding: 12px 16px;
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        color: #2b2d42;
        min-height: 100px;
        margin-top: 10px;
        white-space: pre-line;
    }

    /* ── Grid label styling for form ── */
    .grid-label {
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: #2b2d42;
        margin-bottom: 2px;
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

@st.cache_resource
def load_and_train():
    df = pd.read_csv("merged_output.csv")
    df.drop(columns=['TropI','TropT'], inplace=True)
    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.85]
    df.drop(columns=high_null, inplace=True)
    drop_cols = ['RecordID','Survival','Length_of_stay','completeness']
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
    y_prob = model.predict_proba(X_test)[:,1]
    acc  = accuracy_score(y_test, y_pred)*100
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = cm.ravel()
    return model, acc, auc, len(X_train), len(X_test), tn, fp, fn, tp

model, acc, auc, n_train, n_test, tn, fp, fn, tp = load_and_train()

# ══════════════════════════════════════
# TITLE BAR — exact same as full.py
# ══════════════════════════════════════
st.markdown("""
<div class="title-bar">
  <h2>ICU Mortality Predictor</h2>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# FORM CARD — 2-column grid same as full.py
# ══════════════════════════════════════
st.markdown('<div class="form-card">', unsafe_allow_html=True)

# Each row = 2 fields side by side (label + input)
def row(f1, f2, inputs):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1])
    with c1: st.markdown(f'<div class="grid-label">{f1}</div>', unsafe_allow_html=True)
    with c2: v1 = inputs[f1]()
    with c3: st.markdown(f'<div class="grid-label">{f2}</div>', unsafe_allow_html=True)
    with c4: v2 = inputs[f2]()
    return v1, v2

# Define all input widgets
# Define all input widgets (EMPTY BY DEFAULT)
def mk(key):
    return lambda: st.text_input(
        label=key,
        value="",
        key=key,
        label_visibility="collapsed"
    )
inputs = {
    'SOFA': mk('SOFA'),
    'Age': mk('Age'),
    'GCS': mk('GCS'),
    'BUN': mk('BUN'),
    'Creatinine': mk('Creatinine'),
    'Urine': mk('Urine'),
    'Lactate': mk('Lactate'),
    'pH': mk('pH'),
    'HCO3': mk('HCO3'),
    'MAP': mk('MAP'),
    'HR': mk('HR'),
    'PaO2': mk('PaO2'),
    'FiO2': mk('FiO2'),
    'WBC': mk('WBC'),
    'HCT': mk('HCT'),
    'Platelets': mk('Platelets'),
    'Albumin': mk('Albumin'),
    'Glucose': mk('Glucose'),
    'Weight': mk('Weight'),
}
def mv():
    return st.selectbox(
        'MechVent',
        ["", 1, 0],
        key='MechVent',
        label_visibility="collapsed"
    )

SOFA,      Age       = row('SOFA',       'Age',       inputs)
GCS,       BUN       = row('GCS',        'BUN',       inputs)
Creatinine,Urine     = row('Creatinine', 'Urine',     inputs)
Lactate,   pH        = row('Lactate',    'pH',        inputs)
HCO3,      MAP       = row('HCO3',       'MAP',       inputs)
HR,        PaO2      = row('HR',         'PaO2',      inputs)
FiO2_val = None
WBC_val  = None

# FiO2 + MechVent row
c1,c2,c3,c4 = st.columns([1.2,1,1.2,1])
with c1: st.markdown('<div class="grid-label">FiO2</div>', unsafe_allow_html=True)
with c2: FiO2 = inputs['FiO2']()
with c3: st.markdown('<div class="grid-label">MechVent</div>', unsafe_allow_html=True)
with c4: MechVent = mv()

WBC,       HCT       = row('WBC',        'HCT',       inputs)
Platelets, Albumin   = row('Platelets',  'Albumin',   inputs)
Glucose,   Weight    = row('Glucose',    'Weight',    inputs)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════
# RESULT AREA — always visible like full.py
# ══════════════════════════════════════
st.markdown('<div class="form-card">', unsafe_allow_html=True)

if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None

# Prediction text
if st.session_state.pred_result is None:
    st.markdown('<p class="pred-text" style="color:#2b2d42">Prediction: --</p>', unsafe_allow_html=True)
    st.markdown('<p class="prob-text">Probability: --</p>', unsafe_allow_html=True)
else:
    r = st.session_state.pred_result
    col = "red" if r['pred']==1 else "green"
    txt = "YES (Patient may DIE)" if r['pred']==1 else "NO (Patient will SURVIVE)"
    st.markdown(f'<p class="pred-text" style="color:{col}">Prediction: {txt}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="prob-text">Probability: {r["prob"]*100:.1f}%</p>', unsafe_allow_html=True)

# Risk bar — always shown
needle = 2
if st.session_state.pred_result:
    needle = min(max(st.session_state.pred_result['prob']*100, 1), 99)

st.markdown(f"""
<div class="risk-bar">
  <div class="risk-needle" style="left:{needle}%;"></div>
</div>
<div class="risk-labels">
  <span style="color:green">Low Risk</span>
  <span style="color:orange">Medium</span>
  <span style="color:red">High Risk</span>
</div>
""", unsafe_allow_html=True)

# Conditions box — always shown
cond_text = "Conditions: --"
if st.session_state.pred_result:
    diseases = st.session_state.pred_result['diseases']
    if diseases:
        cond_text = "Conditions:\n" + "\n".join(f"• {d}" for d in diseases)
    else:
        cond_text = "Conditions:\n• No major issues"

st.markdown(f'<div class="cond-box">{cond_text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════
# BUTTONS — Predict & Refresh
# ══════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    predict_clicked = st.button("Predict", use_container_width=True)
with c2:
    refresh_clicked = st.button("Refresh", use_container_width=True)

# ── PREDICT LOGIC ─────────────────────
if predict_clicked:

    values = [
        SOFA, Age, GCS, BUN, Creatinine, Urine, Lactate,
        pH, HCO3, MAP, HR, PaO2, FiO2, MechVent,
        WBC, HCT, Platelets, Albumin, Glucose, Weight
    ]

    # Check if any field is empty
    if "" in values:
        st.error("Please fill all input fields before prediction.")
        st.stop()

    # Convert input values to float
    input_data = [[float(v) for v in values]]

    df_input = pd.DataFrame(input_data, columns=FEATURES)

    prob = model.predict_proba(df_input)[0][1]
    pred = 1 if prob > 0.4 else 0
    user = dict(zip(FEATURES, input_data[0]))

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

    st.session_state.pred_result = {
        'prob': prob,
        'pred': pred,
        'diseases': diseases
    }

    st.rerun()


# ── REFRESH LOGIC ─────────────────────
if refresh_clicked:
    st.session_state.pred_result = None
    st.rerun()

st.markdown("---")
st.caption(
    "⚠️ For educational/research use only. | Random Forest · 20 Vitals · 958 patients"
)