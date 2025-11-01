# app.py — SolarCast Dashboard (Next-15-Min Forecast)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(page_title="🌞 SolarCast — Next-15 Minute Forecast", layout="wide")

# ---------------- Fixed path (your route) ----------------
DATA_PATH = r"D:\DMBI\Solarcast\data\forecast_output_next15.csv"

# ---------------- Helpers ----------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ---------------- Load data ----------------
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"CSV not found at:\n{DATA_PATH}\n\nMake sure the file exists.")
    st.stop()

# Detect datetime column
time_col = "datetime_next15" if "datetime_next15" in df.columns else "datetime"
if time_col not in df.columns:
    st.error("Expected a datetime column: 'datetime_next15' or 'datetime'.")
    st.stop()

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

# Column names (safe)
col_actual  = "actual_next15_kW" if "actual_next15_kW" in df.columns else df.filter(like="actual").columns[0]
col_model   = "pred_model_kW"    if "pred_model_kW"    in df.columns else None
col_base    = "pred_baseline_kW" if "pred_baseline_kW" in df.columns else None

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Controls")

min_date = df[time_col].min().date()
max_date = df[time_col].max().date()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date],
                                   min_value=min_date, max_value=max_date)

# Handle tuple return shape
if isinstance(date_range, (tuple, list)):
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date, end_date = date_range

show_baseline = st.sidebar.checkbox("Show Baseline", True)
show_model    = st.sidebar.checkbox("Show Model", True)
clean_mode    = st.sidebar.checkbox("🧹 Clean presentation mode (Daily + smooth)", value=True)

# ---------------- Filter data ----------------
mask = (df[time_col].dt.date >= start_date) & (df[time_col].dt.date <= end_date)
dfv = df.loc[mask].copy()

# ---------------- Header ----------------
st.title("🌞 SolarCast — Next-15 Minute Forecast")
st.caption("Predicting solar plant output 15 minutes ahead (Random Forest)")

# ---------------- KPIs (computed on filtered window) ----------------
if col_model and col_model in dfv.columns:
    k_mae  = float(np.mean(np.abs(dfv[col_actual] - dfv[col_model])))
    k_rmse = rmse(dfv[col_actual], dfv[col_model])
    k_mape = mape(dfv[col_actual], dfv[col_model])
else:
    k_mae = k_rmse = k_mape = np.nan

k1, k2, k3 = st.columns(3)
k1.metric("MAE (model)",  f"{k_mae:,.2f}" if not np.isnan(k_mae) else "—")
k2.metric("RMSE (model)", f"{k_rmse:,.2f}" if not np.isnan(k_rmse) else "—")
k3.metric("MAPE %",       f"{k_mape:,.2f}" if not np.isnan(k_mape) else "—")

# ---------------- Main chart (sorted + clean mode) ----------------
st.subheader("📉 Actual vs Predicted (Next-15 min)")

# Build plotting frame from filtered data
plot_cols = [time_col, col_actual]
if show_model and col_model:   plot_cols.append(col_model)
if show_baseline and col_base: plot_cols.append(col_base)
dfp = dfv[plot_cols].copy()

if clean_mode:
    # Presentation view: daily mean + 3-day rolling average (smooth & readable)
    dfp = (dfp.set_index(time_col)
              .resample("1D")
              .mean()
              .rolling(window=3, min_periods=1)
              .mean()
              .reset_index())

# ALWAYS sort by time before plotting
dfp = dfp.sort_values(time_col)
line_w = 2.5 if clean_mode else 1.0

# Fresh figure each rerun so checkboxes work correctly
fig, ax = plt.subplots(figsize=(11, 4.5))
handles = []

# Always plot Actual
h_actual, = ax.plot(dfp[time_col], dfp[col_actual], label="Actual", linewidth=line_w)
handles.append(h_actual)

# Conditionally add Baseline / Model
if show_baseline and col_base and col_base in dfp.columns:
    h_base, = ax.plot(dfp[time_col], dfp[col_base], label="Baseline", linestyle=":", linewidth=line_w)
    handles.append(h_base)

if show_model and col_model and col_model in dfp.columns:
    h_model, = ax.plot(dfp[time_col], dfp[col_model], label="Predicted (Random Forest)", linestyle="--", linewidth=line_w)
    handles.append(h_model)

ax.set_xlabel("Datetime")
ax.set_ylabel("Power (kW)")
ax.set_title("Baseline vs Model Forecast Comparison")
ax.legend(handles=handles, loc="upper left")
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# ---------------- Optional sample view (collapsed, sorted) ----------------
with st.expander("🔍 Optional: Sample view (first 100 points)", expanded=False):
    sample = dfp.sort_values(time_col).head(100)
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    hs = []
    hs.append(ax2.plot(sample[time_col], sample[col_actual], label="Actual", linewidth=2)[0])
    if show_model and col_model and col_model in sample.columns:
        hs.append(ax2.plot(sample[time_col], sample[col_model], label="Predicted (Random Forest)", linestyle="--")[0])
    if show_baseline and col_base and col_base in sample.columns:
        hs.append(ax2.plot(sample[time_col], sample[col_base], label="Baseline", linestyle=":")[0])
    ax2.legend(handles=hs)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    plt.close(fig2)

# ---------------- 🔮 Manual What-If Prediction (quick estimator) ----------------
st.subheader("🔮 Manual What-If Prediction (quick estimator)")
st.caption("Enter conditions to estimate next-15-minute power. This is a simple physics-style estimator for demo purposes (not the trained RF).")

c1, c2, c3, c4 = st.columns(4)
irr = c1.number_input("Irradiance (W/m²)", min_value=0, max_value=1200, value=700, step=10)
temp = c2.number_input("Temperature (°C)",  min_value=-20, max_value=60,  value=30, step=1)
humi = c3.slider("Humidity (%)", 0, 100, 50, step=1)
cloud = c4.slider("Cloud cover (%)", 0, 100, 20, step=1)

if st.button("Estimate Power"):
    # Base conversion for a 5 MW site with ~20% overall efficiency:
    base_kW = float(irr) * 1.0  # ≈ irr * (5000 kW * 0.20 / 1000)

    # Temperature derate around 25°C: ~0.3% per °C (bounds)
    temp_factor = 1.0 - 0.003 * (temp - 25)
    temp_factor = float(np.clip(temp_factor, 0.80, 1.05))

    # Humidity light derate around 50%: 0.1% per %RH deviation
    humi_factor = 1.0 - 0.001 * (humi - 50)
    humi_factor = float(np.clip(humi_factor, 0.90, 1.05))

    # Cloud cover impact: ~0.6% per 1% cloud cover (cap at zero)
    cloud_factor = 1.0 - 0.006 * cloud
    cloud_factor = float(np.clip(cloud_factor, 0.0, 1.0))

    est_kW = base_kW * temp_factor * humi_factor * cloud_factor
    st.success(f"Estimated next-15-minute power: **{est_kW:,.2f} kW**")
    with st.expander("See factors used"):
        st.write({
            "base_kW (irr*1.0)": round(base_kW, 2),
            "temp_factor": round(temp_factor, 3),
            "humidity_factor": round(humi_factor, 3),
            "cloud_factor": round(cloud_factor, 3),
        })

# ---------------- Data table ----------------
st.subheader("🧾 Forecast Data (filtered)")
st.dataframe(dfv.head(300))

st.markdown("---")
st.caption(f"📁 Data Source: {DATA_PATH} | Rows in filter: {len(dfv)} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
