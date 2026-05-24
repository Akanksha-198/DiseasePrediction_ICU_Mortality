import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction with ICU Mortality",
    layout="centered"
)

# ─────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────
st.markdown("""
<style>

/* ───────────────── GLOBAL ───────────────── */
.stApp{
    background: linear-gradient(135deg,#eef2ff,#f8fafc,#e0f2fe);
}

.block-container{
    max-width: 950px;
    padding-top: 0rem;
}

/* Hide streamlit default */
#MainMenu, footer, header{
    visibility:hidden;
}

/* ───────────────── TITLE BAR ───────────────── */
.main-title{
    background: linear-gradient(90deg,#6a11cb,#2575fc);
    padding: 20px;
    border-radius: 0px 0px 18px 18px;
    text-align:center;
    margin-bottom: 25px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
}

.main-title h1{
    color:white;
    margin:0;
    font-size:34px;
    font-weight:800;
    font-family:'Segoe UI';
}

.main-title p{
    color:#e9ecef;
    margin-top:5px;
    font-size:14px;
    font-family:'Segoe UI';
}

/* ───────────────── FORM CARD ───────────────── */
.form-card{
    background:white;
    border-radius:20px;
    padding:28px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

/* Labels */
.grid-label{
    font-size:14px;
    font-weight:700;
    color:#1e293b;
    margin-top:8px;
    font-family:'Segoe UI';
}

/* INPUT BOX */
.stTextInput input{
    background:#111827 !important;
    color:white !important;
    border:2px solid #374151 !important;
    border-radius:10px !important;
    padding:10px !important;
    font-size:14px !important;
}

.stTextInput input:focus{
    border:2px solid #7c3aed !important;
    box-shadow:0px 0px 8px rgba(124,58,237,0.5);
}

/* ───────────────── SELECTBOX FIX ───────────────── */
.stSelectbox > div > div{
    background:#111827 !important;
    color:white !important;
    border-radius:10px !important;
    border:2px solid #374151 !important;
}

/* dropdown text */
.stSelectbox div[data-baseweb="select"] > div{
    background:#111827 !important;
    color:white !important;
}

/* make mechvent black */
div[data-baseweb="select"]{
    background:#111827 !important;
    border-radius:10px !important;
}

/* Hide labels */
.stTextInput label,
.stSelectbox label{
    display:none !important;
}

/* ───────────────── BUTTONS ───────────────── */
.stButton button{
    border:none !important;
    border-radius:12px !important;
    padding:12px !important;
    font-size:15px !important;
    font-weight:700 !important;
    transition:0.3s;
}

div[data-testid="column"]:nth-child(1) .stButton button{
    background:linear-gradient(90deg,#00c853,#64dd17) !important;
    color:white !important;
}

div[data-testid="column"]:nth-child(1) .stButton button:hover{
    transform:scale(1.03);
}

div[data-testid="column"]:nth-child(2) .stButton button{
    background:linear-gradient(90deg,#ff416c,#ff4b2b) !important;
    color:white !important;
}

div[data-testid="column"]:nth-child(2) .stButton button:hover{
    transform:scale(1.03);
}

/* ───────────────── RESULT CARD ───────────────── */
.result-card{
    background:white;
    border-radius:20px;
    padding:24px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.08);
}

/* Prediction */
.pred-text{
    text-align:center;
    font-size:24px;
    font-weight:800;
    font-family:'Segoe UI';
}

.prob-text{
    text-align:center;
    font-size:18px;
    color:#334155;
    margin-top:-5px;
}

/* ───────────────── RISK BAR ───────────────── */
.risk-bar{
    width:100%;
    height:40px;
    border-radius:50px;
    background:linear-gradient(to right,#00e676,#ffee58,#ff1744);
    position:relative;
    margin-top:15px;
}

.risk-needle{
    position:absolute;
    top:4px;
    width:32px;
    height:32px;
    border-radius:50%;
    background:#111827;
    border:3px solid white;
    transform:translateX(-50%);
    box-shadow:0px 0px 10px rgba(0,0,0,0.4);
}

.risk-labels{
    display:flex;
    justify-content:space-between;
    font-size:13px;
    margin-top:6px;
    font-weight:700;
}

/* ───────────────── CONDITIONS BOX ───────────────── */
.cond-box{
    background:#f1f5f9;
    border-left:6px solid #7c3aed;
    padding:18px;
    border-radius:12px;
    margin-top:18px;
    font-size:14px;
    font-family:'Segoe UI';
    color:#1e293b;
    white-space:pre-line;
}

/* Footer */
.footer{
    text-align:center;
    color:#64748b;
    font-size:13px;
    margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

# ───────────────── FEATURES ─────────────────
FEATURES = [
    'SOFA','Age','GCS','BUN','Creatinine','Urine','Lactate',
    'pH','HCO3','MAP','HR','PaO2','FiO2','MechVent',
    'WBC','HCT','Platelets','Albumin','Glucose','Weight'
]

TARGET = 'In-hospital_death'

# ───────────────── MODEL ─────────────────
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

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = cm.ravel()

    return model, acc, auc, tn, fp, fn, tp

model, acc, auc, tn, fp, fn, tp = load_and_train()

# ───────────────── TITLE ─────────────────
st.markdown("""
<div class="main-title">
    <h1> Disease Prediction with ICU Mortality</h1>
</div>
""", unsafe_allow_html=True)

# ───────────────── FORM CARD ─────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)

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
        ["",1,0],
        key='MechVent',
        label_visibility="collapsed"
    )

def row(f1,f2,inputs):

    c1,c2,c3,c4 = st.columns([1.2,1,1.2,1])

    with c1:
        st.markdown(f'<div class="grid-label">{f1}</div>', unsafe_allow_html=True)

    with c2:
        v1 = inputs[f1]()

    with c3:
        st.markdown(f'<div class="grid-label">{f2}</div>', unsafe_allow_html=True)

    with c4:
        v2 = inputs[f2]()

    return v1,v2

SOFA, Age = row('SOFA','Age',inputs)
GCS, BUN = row('GCS','BUN',inputs)
Creatinine, Urine = row('Creatinine','Urine',inputs)
Lactate, pH = row('Lactate','pH',inputs)
HCO3, MAP = row('HCO3','MAP',inputs)
HR, PaO2 = row('HR','PaO2',inputs)

# FiO2 + MechVent
c1,c2,c3,c4 = st.columns([1.2,1,1.2,1])

with c1:
    st.markdown('<div class="grid-label">FiO2</div>', unsafe_allow_html=True)

with c2:
    FiO2 = inputs['FiO2']()

with c3:
    st.markdown('<div class="grid-label">MechVent</div>', unsafe_allow_html=True)

with c4:
    MechVent = mv()

WBC, HCT = row('WBC','HCT',inputs)
Platelets, Albumin = row('Platelets','Albumin',inputs)
Glucose, Weight = row('Glucose','Weight',inputs)

st.markdown('</div>', unsafe_allow_html=True)

# ───────────────── RESULT CARD ─────────────────
st.markdown('<div class="result-card">', unsafe_allow_html=True)

if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None

if st.session_state.pred_result is None:

    st.markdown(
        '<p class="pred-text" style="color:#334155">Prediction: --</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="prob-text">Probability: --</p>',
        unsafe_allow_html=True
    )

else:

    r = st.session_state.pred_result

    color = "red" if r['pred']==1 else "green"

    txt = (
        "YES (High Mortality Risk)"
        if r['pred']==1
        else
        "NO (Patient Likely Stable)"
    )

    st.markdown(
        f'<p class="pred-text" style="color:{color}">{txt}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="prob-text">Mortality Probability: {r["prob"]*100:.2f}%</p>',
        unsafe_allow_html=True
    )

# Risk bar
needle = 2

if st.session_state.pred_result:
    needle = min(max(st.session_state.pred_result['prob']*100,1),99)

st.markdown(f"""
<div class="risk-bar">
    <div class="risk-needle" style="left:{needle}%"></div>
</div>

<div class="risk-labels">
    <span style="color:green;">LOW RISK</span>
    <span style="color:orange;">MEDIUM</span>
    <span style="color:red;">HIGH RISK</span>
</div>
""", unsafe_allow_html=True)

# Conditions
cond_text = "Conditions: --"

if st.session_state.pred_result:

    diseases = st.session_state.pred_result['diseases']

    if diseases:
        cond_text = "Detected Conditions:\\n" + "\\n".join(
            f"• {d}" for d in diseases
        )
    else:
        cond_text = "Detected Conditions:\\n• No major abnormalities"

st.markdown(
    f'<div class="cond-box">{cond_text}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ───────────────── BUTTONS ─────────────────
c1,c2 = st.columns(2)

with c1:
    predict_clicked = st.button(
        "🔍 Predict",
        use_container_width=True
    )

with c2:
    refresh_clicked = st.button(
        "🔄 Refresh",
        use_container_width=True
    )

# ───────────────── PREDICT LOGIC ─────────────────
if predict_clicked:

    values = [
        SOFA, Age, GCS, BUN, Creatinine, Urine,
        Lactate, pH, HCO3, MAP, HR, PaO2,
        FiO2, MechVent, WBC, HCT,
        Platelets, Albumin, Glucose, Weight
    ]

    if "" in values:

        st.error("⚠️ Please fill all input fields.")
        st.stop()

    input_data = [[float(v) for v in values]]

    df_input = pd.DataFrame(
        input_data,
        columns=FEATURES
    )

    prob = model.predict_proba(df_input)[0][1]

    pred = 1 if prob > 0.4 else 0

    user = dict(zip(FEATURES,input_data[0]))

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

# ───────────────── REFRESH ─────────────────
if refresh_clicked:

    st.session_state.pred_result = None
    st.rerun()

# ───────────────── FOOTER ─────────────────
st.markdown(f"""
<div class="footer">
⚠️ Educational / Research Purpose Only <br>
Random Forest Model • Accuracy: {acc:.2f}% • AUC: {auc:.2f}
</div>
""", unsafe_allow_html=True)