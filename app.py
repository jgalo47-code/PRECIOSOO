import streamlit as st
from modules import data_processing as dp, exploratory_analysis as eda, prediction, dashboard

st.set_page_config(page_title="Campus Facility Reservation Monitor", layout="wide")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption(f"Model {dp.META['version']}")

if menu == "Home":
    st.title("Campus Facility Reservation Monitoring")
    st.write("Predicts whether a reservation will be **Completed** or **Not Completed** (cancelled/no-show) to help administrators plan facility use. "
             "Reuses the midterm Decision Tree, saved as a preprocessing + model pipeline (loaded from joblib, never retrained here).")
    st.caption(f"Data in use: {dp.source_name()}. Differences from midterm: categorical encoding moved inside the pipeline (one-hot); metrics are weighted averages.")
    st.info("Predictions are decision support, not authorization to reject a reservation. Uses synthetic data.")
elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try: df = dp.read_csv(up)
        except Exception as e: st.error(f"Could not read file: {e}"); st.stop()
        errs, warns = dp.validate(df)
        for e in errs: st.error(e)
        for w in warns: st.warning(w)
        if not errs:
            st.session_state["df"] = df; st.success("Valid schema. Data is now used across all modules.")
            st.dataframe(df.head(20)); st.write("Rows:", len(df), "| Columns:", df.shape[1])
            info, dups = dp.quality_summary(df); st.dataframe(info); st.write("Duplicate rows:", dups)
    else:
        st.caption("No upload yet: bundled sample data is used. Required columns: " + ", ".join(dp.FEATURES))
elif menu == "Data Analysis": eda.show()
elif menu == "Prediction": prediction.show()
elif menu == "Dashboard": dashboard.show()
else:
    st.header("System Design (SAD Integration)")
    st.subheader("Functional requirements")
    st.markdown("1. FR-DM-01: Display reservation counts per facility type.\n2. FR-DM-02: Display cancellation/no-show % per user category with reservation totals.\n"
                "3. FR-DM-03: Display predicted risk levels and prediction information for administrator review.\n4. FR-04: Upload and validate reservation CSV (schema, nulls, duplicates).\n"
                "5. FR-05: Let administrators record the final decision on flagged reservations.")
    st.subheader("Non-functional requirements")
    st.markdown("1. NFR1: Page responses under 3 s (model cached).\n2. NFR2: No personal student data; synthetic/authorized data only.\n"
                "3. NFR3: Model version, timestamp, and limitations always visible.")
    st.subheader("Actors")
    st.markdown("- **Administrator**: uploads data, views statistics and predicted risk, records final decisions.\n"
                "- **User** (student/organization/staff): manages own reservations; cannot see predictions.")
    st.image("assets/use_case_diagram.png", caption="Updated use case diagram")
    st.subheader("Decision workflow")
    st.code("Reservation data -> Validate -> Predict (saved Decision Tree pipeline) -> Risk level + follow-up\n"
            "-> Administrator review queue -> Human decision recorded -> Reservation outcome", language="text")
