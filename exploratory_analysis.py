import streamlit as st, plotly.express as px
from modules.data_processing import get_active_df, source_name
from modules.features import TARGET, NUM, CAT

def show():
    st.header("Exploratory Data Analysis"); st.caption(f"Data: {source_name()}")
    df = get_active_df(); has_t = TARGET in df.columns
    st.subheader("Descriptive statistics"); st.dataframe(df.describe(include="all").T)
    if has_t:
        st.subheader("Class distribution")
        vc = df[TARGET].value_counts().reset_index(); vc.columns = ["outcome", "count"]
        st.plotly_chart(px.pie(vc, names="outcome", values="count"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        f = st.selectbox("Reservation count by", CAT)
        r = df[f].value_counts().reset_index(); r.columns = [f, "count"]
        st.plotly_chart(px.bar(r, x=f, y="count", title=f"Reservations by {f}"), use_container_width=True)
    with c2:
        if has_t:
            g = st.selectbox("Cancellation/no-show % by", CAT + ["Lead_Time_Category"])
            t = df.groupby(g, observed=True)[TARGET].agg(rate=lambda s: (s == "Not Completed").mean()*100, n="size").reset_index()
            st.plotly_chart(px.bar(t, x=g, y="rate", hover_data=["n"], labels={"rate": "Not Completed %"}), use_container_width=True)
    x = st.selectbox("Distribution of", NUM)
    st.plotly_chart(px.histogram(df, x=x, color=TARGET if has_t else None, barmode="overlay"), use_container_width=True)
    a, b = st.columns(2); xa = a.selectbox("Scatter X", NUM, 0); yb = b.selectbox("Scatter Y", NUM, 1)
    st.plotly_chart(px.scatter(df, x=xa, y=yb, color=TARGET if has_t else None, opacity=.6), use_container_width=True)
