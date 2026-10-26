import json, os, pandas as pd, streamlit as st
from modules.features import FEATURES, NUM, TARGET, add_derived

META = json.load(open("models/metrics.json"))
REAL = "data/Facility_reservation_cleaned.csv"; SAMPLE = "data/sample_reservations.csv"

@st.cache_data
def read_csv(file): return pd.read_csv(file)

def validate(df):
    errors, warnings = [], []
    if df.empty: return ["The file contains no rows."], warnings
    missing = [c for c in FEATURES if c not in df.columns]
    if missing: return [f"Missing required columns: {', '.join(missing)}"], warnings
    bad = [c for c in NUM if not pd.api.types.is_numeric_dtype(df[c])]
    if bad: errors.append(f"Columns must be numeric: {', '.join(bad)}")
    elif (df[NUM] < 0).any().any(): warnings.append("Negative values found in numeric columns; investigate before use.")
    if TARGET not in df.columns and "Usage_Status" not in df.columns:
        warnings.append("No Reservation_Outcome/Usage_Status column: observed statistics unavailable; predictions still work.")
    return errors, warnings

def quality_summary(df):
    return pd.DataFrame({"nulls": df.isna().sum(), "dtype": df.dtypes.astype(str)}), int(df.duplicated().sum())

def source_name():
    return "uploaded file" if st.session_state.get("df") is not None else ("midterm cleaned dataset" if os.path.exists(REAL) else "synthetic sample data")

def get_active_df():
    df = st.session_state.get("df")
    if df is None: df = pd.read_csv(REAL if os.path.exists(REAL) else SAMPLE)
    return add_derived(df)
