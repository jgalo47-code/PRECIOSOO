"""Shared, dependency-light feature logic (also used by train_model.py)."""
import pandas as pd

NUM = ["Participants", "Lead_Time_Days", "Reservation_Duration"]
CAT = ["Facility_Type", "User_Category"]
FEATURES = CAT + NUM            # only values known at submission time (no leakage)
TARGET = "Reservation_Outcome"  # Completed / Not Completed
POSITIVE = "Not Completed"      # cancellation / no-show

def add_derived(df):
    df = df.copy()
    for c in CAT:
        if c in df: df[c] = df[c].astype(str).str.strip().str.title()
    if "Lead_Time_Days" in df:
        df["Lead_Time_Category"] = pd.cut(pd.to_numeric(df["Lead_Time_Days"], errors="coerce"), [-1, 2, 7, 14, float("inf")],
                                          labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"])
    if TARGET not in df and "Usage_Status" in df:   # midterm rule
        m = {"used": "Completed", "cancelled": "Not Completed", "no-show": "Not Completed", "no show": "Not Completed"}
        df[TARGET] = df["Usage_Status"].astype(str).str.strip().str.lower().map(m)
    return df
