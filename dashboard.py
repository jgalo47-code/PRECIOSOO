import datetime as dt, pandas as pd, streamlit as st, plotly.express as px
from modules.data_processing import get_active_df, source_name
from modules.features import TARGET, FEATURES
from modules.prediction import predict_df, risk_band

ACTIONS = {"High": "Contact requester to confirm; consider waitlist.", "Medium": "Send reminder before the date.", "Low": "No action needed."}

def show():
    st.header("Facility Administrator Dashboard"); st.caption(f"Data: {source_name()}")
    df = get_active_df().copy(); has_t = TARGET in df.columns
    st.subheader("Observed historical statistics")
    k = st.columns(3); k[0].metric("Total reservations", len(df))
    if has_t: k[1].metric("Not Completed %", f"{(df[TARGET] == 'Not Completed').mean():.1%}")
    c1, c2 = st.columns(2)
    r = df.Facility_Type.value_counts().reset_index(); r.columns = ["Facility_Type", "count"]
    c1.plotly_chart(px.bar(r, x="Facility_Type", y="count", title="Reservations by facility type"), use_container_width=True)
    if has_t:
        t = df.groupby("User_Category")[TARGET].agg(not_completed_pct=lambda s: round((s == "Not Completed").mean()*100, 2), reservations="size").reset_index()
        c2.plotly_chart(px.bar(t, x="User_Category", y="not_completed_pct", hover_data=["reservations"], title="Cancellation/no-show % by user category"), use_container_width=True)
        st.dataframe(t)
        o = df.groupby(["Lead_Time_Category", TARGET], observed=True).size().reset_index(name="count")
        st.plotly_chart(px.bar(o, x="Lead_Time_Category", y="count", color=TARGET, barmode="group", title="Outcomes by lead-time category"), use_container_width=True)
    st.divider(); st.subheader("Model predictions (not facts)")
    try: _, p = predict_df(df)
    except Exception as e: st.error(f"Cannot score data: {e}"); return
    df["risk_score"] = p.round(3); df["risk_level"] = [risk_band(x) for x in p]; df["recommended_follow_up"] = df.risk_level.map(ACTIONS)
    st.metric("High-risk reservations", int((df.risk_level == "High").sum()))
    st.plotly_chart(px.histogram(df, x="risk_level", category_orders={"risk_level": ["Low", "Medium", "High"]}), use_container_width=True)
    st.markdown("**Review queue (top 20 by predicted risk).** Main drivers: see top predictors on the Prediction page. Predictions are not guaranteed.")
    q = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(q[FEATURES + ["risk_score", "risk_level", "recommended_follow_up"]])
    st.subheader("Record final human decision")
    log = st.session_state.setdefault("decisions", []); c = st.columns(3)
    idx = c[0].selectbox("Queue row", q.index.tolist()); dec = c[1].selectbox("Decision", ["Approve/keep", "Send reminder", "Contact requester", "Release slot"])
    if c[2].button("Save decision"): log.append({"row": idx, "decision": dec, "by": "Administrator", "at": f"{dt.datetime.now():%Y-%m-%d %H:%M}"})
    if log: st.table(pd.DataFrame(log))
