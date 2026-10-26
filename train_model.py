"""Rebuild the midterm Decision Tree as a saved pipeline. Run ONCE locally, then commit models/.
Put your cleaned midterm CSV at data/Facility_reservation_cleaned.csv (same 80/20 split, random_state=42).
Without it, a synthetic sample with the same columns is generated so the app still runs."""
import json, os, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from modules.features import *

REAL = "data/Facility_reservation_cleaned.csv"
if os.path.exists(REAL):
    df, src = add_derived(pd.read_csv(REAL).drop_duplicates()), "midterm cleaned dataset"
else:
    rng = np.random.default_rng(42); n = 400
    df = pd.DataFrame({"Facility_Type": rng.choice(["Auditorium", "Gymnasium", "Classroom", "Laboratory", "Function Hall"], n, p=[.24, .2, .22, .16, .18]),
                       "User_Category": rng.choice(["Student", "Organization", "Faculty", "Staff"], n),
                       "Participants": rng.integers(5, 300, n), "Lead_Time_Days": rng.integers(0, 45, n),
                       "Reservation_Duration": rng.integers(1, 9, n)})
    z = -.6 + .03*df.Lead_Time_Days + .5*(df.User_Category == "Organization")
    df[TARGET] = np.where(rng.random(n) < 1/(1+np.exp(-z)) , "Not Completed", "Completed")
    df.to_csv("data/sample_reservations.csv", index=False); df = add_derived(df); src = "synthetic sample (replace with midterm data)"
df = df.dropna(subset=FEATURES + [TARGET]); df = df[df[TARGET].isin(["Completed", "Not Completed"])]
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGET], test_size=.2, random_state=42)
pipe = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")),
                 ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))])
pipe.fit(Xtr, ytr); p = pipe.predict(Xte)
pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
names = pipe.named_steps["prep"].get_feature_names_out(); imp = pipe.named_steps["clf"].feature_importances_
top = [[str(n).split("__")[-1], round(float(v), 3)] for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:5]]
m = dict(version="v1.0-decision-tree", data_source=src, accuracy=accuracy_score(yte, p), precision=pr, recall=rc, f1=f1,
         n_train=len(Xtr), n_test=len(Xte), top_predictors=top,
         midterm_reported=dict(accuracy=.412, precision=.484, recall=.412, f1=.430))
joblib.dump(pipe, "models/trained_model.joblib"); json.dump(m, open("models/metrics.json", "w"), indent=2); print(m)
