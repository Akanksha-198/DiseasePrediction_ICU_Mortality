import pandas as pd
import tkinter as tk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ── FEATURES ──────────────────────────
FEATURES = [
    'SOFA','Age','GCS','BUN','Creatinine','Urine','Lactate',
    'pH','HCO3','MAP','HR','PaO2','FiO2','MechVent',
    'WBC','HCT','Platelets','Albumin','Glucose','Weight'
]

TARGET = 'In-hospital_death'

# ══════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════
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

print("=" * 55)
print("   ICU MORTALITY PREDICTOR — MODEL TRAINING")
print("=" * 55)
print(f"Total patients  : {len(df)}")
print(f"Survived (0)    : {(y == 0).sum()}  ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"Died     (1)    : {(y == 1).sum()}  ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"Features used   : {len(FEATURES)}")

# ══════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT  (80/20)
# ══════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test
    random_state=42,
    stratify=y           # maintain class balance in both splits
)

print(f"\n{'─' * 55}")
print(f"  TRAIN / TEST SPLIT  (80% / 20%)")
print(f"{'─' * 55}")
print(f"  Training patients : {len(X_train)}  (80%)")
print(f"  Testing  patients : {len(X_test)}  (20%)")

# ══════════════════════════════════════
# STEP 3 — TRAIN MODEL (only on X_train)
# ══════════════════════════════════════
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)   # ← only training data, not full dataset

# ══════════════════════════════════════
# STEP 4 — EVALUATE on X_test (unseen)
# ══════════════════════════════════════
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred) * 100
auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'─' * 55}")
print(f"  MODEL EVALUATION RESULTS  (on test data only)")
print(f"{'─' * 55}")
print(f"  Accuracy         : {acc:.2f}%")
print(f"  AUC-ROC          : {auc:.4f}")
print(f"  Sensitivity      : {tp/(tp+fn)*100:.2f}%  (deaths caught)")
print(f"  Specificity      : {tn/(tn+fp)*100:.2f}%  (survivors correct)")
print(f"\n  Confusion Matrix :")
print(f"    True  Negative (Survived→Survived) : {tn}")
print(f"    False Positive (Survived→Died)     : {fp}")
print(f"    False Negative (Died→Survived)     : {fn}")
print(f"    True  Positive (Died→Died)         : {tp}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Survived', 'Died']))
print("=" * 55)
print("  Model ready — GUI loading...")
print("=" * 55)

# ── COLORS ────────────────────────────
BG      = "#F7F9FC"
CARD    = "#FFFFFF"
PRIMARY = "#A8DADC"
ACCENT  = "#FFC8DD"
TEXT    = "#2b2d42"

# ── GUI ───────────────────────────────
root = tk.Tk()
root.title("ICU Mortality Predictor")
root.geometry("820x880")
root.configure(bg=BG)

# ── TITLE ─────────────────────────────
tk.Label(root, text="ICU Mortality Predictor",
         font=("Segoe UI", 20, "bold"),
         bg=PRIMARY, fg=TEXT, pady=12).pack(fill="x")

# ── ACCURACY BANNER ───────────────────
tk.Label(root,
         text=f"Model: Random Forest  |  Train: {len(X_train)} patients  |  Test: {len(X_test)} patients  |  Accuracy: {acc:.1f}%  |  AUC: {auc:.3f}",
         font=("Segoe UI", 9),
         bg="#d0f0f0", fg="#1a4a4a", pady=4).pack(fill="x")

frame = tk.Frame(root, bg=CARD)
frame.pack(pady=15, padx=20)

entries = {}

# ── INPUT GRID ────────────────────────
for i, f in enumerate(FEATURES):
    row = i // 2
    col = i % 2

    tk.Label(frame, text=f,
             bg=CARD, fg=TEXT,
             font=("Segoe UI", 10, "bold")
    ).grid(row=row, column=col*2, padx=10, pady=6, sticky="w")

    e = tk.Entry(frame, width=12, bg="#eef2f3")
    e.grid(row=row, column=col*2+1, padx=10, pady=6)

    entries[f] = e

# ── RESULT SECTION ────────────────────
result_frame = tk.Frame(root, bg=CARD)
result_frame.pack(pady=15, padx=20, fill="x")

status_label = tk.Label(result_frame,
                        text="Prediction: --",
                        font=("Segoe UI", 16, "bold"),
                        bg=CARD, fg=TEXT)
status_label.pack()

prob_label = tk.Label(result_frame,
                      text="Probability: --",
                      font=("Segoe UI", 12),
                      bg=CARD)
prob_label.pack()

# ── SCALE BAR (GREEN → RED) ───────────
canvas = tk.Canvas(result_frame,
                   width=400, height=40,
                   bg=CARD, highlightthickness=0)
canvas.pack(pady=10)

for i in range(400):
    if i < 200:
        r = int(255 * (i / 200))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (i - 200) / 200))
    color = f'#{r:02x}{g:02x}00'
    canvas.create_line(i, 0, i, 40, fill=color)

indicator = canvas.create_oval(0, 5, 20, 35, fill="black")

# ── SCALE LABELS ──────────────────────
label_frame = tk.Frame(result_frame, bg=CARD)
label_frame.pack()

tk.Label(label_frame, text="Low Risk",
         bg=CARD, fg="green").pack(side="left", padx=10)
tk.Label(label_frame, text="Medium",
         bg=CARD, fg="orange").pack(side="left", padx=130)
tk.Label(label_frame, text="High Risk",
         bg=CARD, fg="red").pack(side="right", padx=10)

# ── CONDITIONS BOX ────────────────────
condition_box = tk.Label(result_frame,
                         text="Conditions: --",
                         bg="#eaf4f4",
                         width=65, height=5,
                         justify="left",
                         font=("Segoe UI", 10))
condition_box.pack(pady=10)

# ── FUNCTIONS ─────────────────────────
def predict():
    try:
        data = []
        user = {}

        for f in FEATURES:
            val = float(entries[f].get())
            data.append(val)
            user[f] = val

        df_input = pd.DataFrame([data], columns=FEATURES)

        prob = model.predict_proba(df_input)[0][1]
        pred = 1 if prob > 0.4 else 0

        result = "YES (Patient may DIE)" if pred == 1 else "NO (Patient will SURVIVE)"

        status_label.config(
            text=f"Prediction: {result}",
            fg="red" if pred == 1 else "green"
        )

        prob_label.config(text=f"Probability: {prob * 100:.1f}%")

        pos = int(prob * 380)
        canvas.coords(indicator, pos, 5, pos + 20, 35)

        # CONDITIONS
        diseases = []

        if user['Lactate'] > 2 and user['MAP'] < 65:
            diseases.append("Shock")

        if user['Creatinine'] > 1.5 and user['BUN'] > 25:
            diseases.append("Kidney Failure")

        if user['GCS'] < 8:
            diseases.append("Neurological Issue")

        if user['PaO2'] < 60 or user['FiO2'] > 0.6:
            diseases.append("Respiratory Failure")

        if user['WBC'] > 12000:
            diseases.append("Infection")

        if diseases:
            text = "\n".join(f"• {d}" for d in diseases)
        else:
            text = "• No major issues"

        condition_box.config(text="Conditions:\n" + text)

    except:
        status_label.config(text="⚠️ Invalid Input", fg="red")


def clear_fields():
    for e in entries.values():
        e.delete(0, tk.END)

    status_label.config(text="Prediction: --", fg=TEXT)
    prob_label.config(text="Probability: --")
    condition_box.config(text="Conditions: --")
    canvas.coords(indicator, 0, 5, 20, 35)


# ── BUTTONS ───────────────────────────
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict",
          command=predict,
          bg=PRIMARY, width=12,
          font=("Segoe UI", 11, "bold")
          ).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Refresh",
          command=clear_fields,
          bg=ACCENT, width=12,
          font=("Segoe UI", 11, "bold")
          ).grid(row=0, column=1, padx=10)

root.mainloop()