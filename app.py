"""Sri Lankan Food Price Forecasting - Ensemble Learning Dashboard.

Run:  streamlit run app.py
Needs the files created by the notebook in ./models and foodprice.py next to this file.
Generated with the help of Claude (Anthropic); see the disclosure in the notebook.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import foodprice as fp
from foodprice import FAMILY, derive, to_price

st.set_page_config(page_title="LK Food Price Forecasting", page_icon="🌾", layout="wide")
COLORS = {"Baseline": "#79827C", "Bagging": "#477A91", "Boosting": "#AE693B",
          "Stacking": "#347158", "Hybrid": "#826894"}


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

@st.cache_resource
def load_models():
    return joblib.load(MODEL_DIR / "models.joblib")


@st.cache_data
def load_tables():
    res = pd.read_csv(MODEL_DIR / "results.csv", index_col=0)
    bycat = pd.read_csv(MODEL_DIR / "by_category.csv", index_col=0)
    imp = pd.read_csv(MODEL_DIR / "importance.csv", index_col=0)
    preds = pd.read_csv(MODEL_DIR / "test_predictions.csv.gz")
    panel = pd.read_csv(MODEL_DIR / "panel.csv.gz")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    with (MODEL_DIR / "summary.json").open() as fh:
        summ = json.load(fh)
    return res, bycat, imp, preds, panel, summ


bundle = load_models()
models, cats = bundle["models"], bundle["cats"]
res, bycat, imp, preds, panel, summ = load_tables()

st.html(ROOT / ".streamlit" / "dashboard.css")
with st.container(key="dashboard-heading", gap=20):
    st.title("🌾 Sri Lankan Food Price Forecasting")
    st.caption("Bagging, boosting, stacking and hybrid models on WFP Sri Lankan retail food prices (2023–2025). Historical model evaluation, not live prices.")
page = st.sidebar.radio("Page", ["Overview", "Model comparison", "Forecast explorer", "Why it works"])

# ------------------------------------------------------------------ Overview
if page == "Overview":
    best = summ["best"]
    naive = res.loc["Naive (last month)", "MAE (LKR)"]
    test_months = pd.PeriodIndex(preds['month'], freq='M')
    with st.grid(4, min_column_width=240):
        st.metric("Series modelled", f"{panel.groupby(['market', 'commodity']).ngroups:,}", height="stretch")
        st.metric("Best model (MAE)", best, height="stretch")
        st.metric(
            "Its MAE",
            f"{res.loc[best, 'MAE (LKR)']:.1f} LKR",
            f"{(res.loc[best, 'MAE (LKR)'] / naive - 1) * 100:.1f}% vs naive",
            delta_color="inverse",
            delta_arrow="off",
            height="stretch",
        )
        st.metric(
            "Test window",
            f"{test_months.min().strftime('%b %Y')}–{test_months.max().strftime('%b %Y')}",
            height="stretch",
        )
    st.container(key="overview-notes").markdown(f"""
**The problem.** Forecast the following month's retail price in Sri Lankan markets (LKR per reported unit: usually kg, sometimes a single item).

**Why it is hard.** Food prices are persistent, so the *same price as last month* baseline is strong. A useful model needs to anticipate small one-month changes.

**What I compared.** Bagging (Random Forest), boosting (GBM, XGBoost, CatBoost), stacking and two hybrids (RF+ARIMA, RF+ANN), compared with naive / ARIMA / Ridge baselines.

**Main result on this held-out period.** **{best}** has the lowest MAE. Its test-row-weighted mean absolute-error improvement over the naive baseline is **{summ['gain']:.2f} LKR per prediction**. An exploratory month-block bootstrap gives a 95% interval of **{summ['ci'][0]:.2f} to {summ['ci'][1]:.2f} LKR**. There are just six held-out months, so this interval should not be interpreted as conclusive statistical evidence. The outcome could change in another period.

**Forecast date.** This historical dataset ends in September 2025; the next-month explorer forecasts **October 2025**. It is not a live November 2026 forecast.
""")

# ------------------------------------------------------- Model comparison
elif page == "Model comparison":
    st.subheader("Test-set accuracy (Apr–Sep 2025, never seen in training)")
    metric = st.selectbox("Metric", ["MAE (LKR)", "RMSE (LKR)", "MAPE (%)", "R2", "Direction acc. (%)"])
    r = res.sort_values(metric, ascending=(metric not in ["R2", "Direction acc. (%)"]))
    fig = px.bar(r.reset_index(), x=metric, y="index", color="Family", orientation="h",
                 color_discrete_map=COLORS, labels={"index": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending" if metric in ["R2", "Direction acc. (%)"] else "total descending"},
                      height=430, legend_title="")
    lo, hi = r[metric].min(), r[metric].max()
    pad = (hi - lo) * 0.25
    fig.update_xaxes(range=[lo - pad, hi + pad * 0.3])
    fig.add_vline(x=res.loc["Naive (last month)", metric], line_dash="dash", annotation_text="naive")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(res.drop(columns=["Train time (s)"]).style.format(precision=3), use_container_width=True)
    st.subheader("Accuracy by food category (MAE, LKR per reported unit)")
    st.caption("Category-level performance can differ from the overall ranking; the data mix kg and per-item prices.")
    st.dataframe(bycat.style.format(precision=1), use_container_width=True)
    w = summ["stack_weights"]
    st.info("Stacking meta-model weights: " + ", ".join(f"{k}: {v}" for k, v in w.items()))

# --------------------------------------------------------- Forecast explorer
elif page == "Forecast explorer":
    st.subheader("Pick a food item and market")
    # Only allow a series when all required historical lag values exist for
    # the real next month after the dataset's final observed month.
    forecast_month = panel.month.max()
    eligible = panel.loc[panel.month == forecast_month, ['market', 'commodity']].drop_duplicates()
    if eligible.empty:
        st.error("No series has sufficient recent history for a next-month forecast.")
        st.stop()
    st.caption(f"Historical forecast for {forecast_month}: only {len(eligible)} series with complete recent monthly prices are selectable.")
    c1, c2 = st.columns(2)
    com = c1.selectbox("Commodity", sorted(eligible.commodity.unique()))
    mks = sorted(eligible[eligible.commodity == com].market.unique())
    mk = c2.selectbox("Market", mks)
    sel = panel[(panel.commodity == com) & (panel.market == mk)].sort_values("t")
    unit = sel.unit.iloc[-1]
    unit_label = 'kg' if unit == 'KG' else ('item' if unit == 'Unit' else unit)
    hist = sel[sel.obs]
    show = st.multiselect("Models to show on the chart", list(models),
                          default=["Naive (last month)", "Random Forest", "Stacking"])
    pr = preds[(preds.commodity == com) & (preds.market == mk)]
    fig = go.Figure()
    fig.add_scatter(x=hist.month.astype(str), y=hist.price, name="Actual", line=dict(color="#233E33", width=3))
    for n in show:
        fig.add_scatter(x=pr.month, y=pr[n], name=n, mode="lines+markers")
    fig.update_layout(height=420, yaxis_title=f"LKR per {unit_label}", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Coloured lines are one-step-ahead predictions on the 6 test months (models trained only on earlier data).")

    st.subheader(f"Forecast for {forecast_month} (month after the dataset ends)")
    fut = sel.loc[sel.month == forecast_month].iloc[[-1]].copy()
    st.markdown("**What-if:** edit the last three observed prices and see how each model reacts.")
    k1, k2, k3 = st.columns(3)
    p1 = k1.number_input(f"Last month (LKR/{unit_label})", value=float(round(np.exp(fut.lp1.iloc[0]), 2)), min_value=1.0)
    p2 = k2.number_input(f"2 months ago (LKR/{unit_label})", value=float(round(np.exp(fut.lp2.iloc[0]), 2)), min_value=1.0)
    p3 = k3.number_input(f"3 months ago (LKR/{unit_label})", value=float(round(np.exp(fut.lp3.iloc[0]), 2)), min_value=1.0)
    fut["lp1"], fut["lp2"], fut["lp3"] = np.log(p1), np.log(p2), np.log(p3)
    fut = derive(fut)
    out = pd.DataFrame({n: to_price(fut, m.predict(fut))[0] for n, m in models.items()}, index=["Forecast (LKR)"]).T
    out["Change vs last month"] = (out["Forecast (LKR)"] / p1 - 1) * 100
    out["Family"] = out.index.map(FAMILY)
    st.dataframe(out.style.format({"Forecast (LKR)": "{:.2f}", "Change vs last month": "{:+.1f}%"}),
                 use_container_width=True)
    st.caption(f"All models were refit through September 2025. Prices shown are LKR per {unit_label}. These are historical next-month model estimates, not guaranteed or live market prices. What-if edits do not recompute national ARIMA forecasts.")

# ------------------------------------------------------------ Why it works
else:
    st.subheader("What the Random Forest relies on")
    fig = px.bar(imp.sort_values("importance"), x="importance", orientation="h")
    fig.update_layout(height=420, yaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**How to interpret the results**
- **Bagging.** Random Forest averages predictions from trees to reduce variance. This may help with noisy month-to-month changes; the observed MAE/RMSE table determines whether it actually helped here.
- **Boosting.** GBM, XGBoost and CatBoost successively learn corrections. They can model complicated effects but may need different tuning for volatile commodities. We cannot prove overfitting from one evaluation window.
- **Stacking.** The second-stage Ridge model combines base learners using predictions from forward-chaining folds. It may not beat the strongest individual model if their errors are similar.
- **Hybrids.** RF+ARIMA models national changes plus market-specific deviations. RF+ANN blends forest and neural-network estimates. Their relative merit should be judged by the held-out scores, not the method name.
- **Feature importance.** A high importance indicates which inputs the forest used to split trees; it does **not** establish a causal economic relationship.
- **Limitations.** Test results cover April–September 2025 only, with a limited training period, incomplete series excluded rather than interpolated, no exogenous drivers, untuned hyperparameters, and differing commodity price units. Results may differ under another evaluation period.
""")
