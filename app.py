# app.py — SolarCast Dashboard with Manual Prediction
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import json
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ---------------- Page config ----------------
st.set_page_config(
    page_title="🌞 SolarCast — Solar Power Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations and modern styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ECDC4;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        min-width: 250px !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 0.5rem;
    }
    
    /* Make sidebar always visible */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 250px !important;
    }
    
    /* Add prominent button for manual prediction */
    .manual-pred-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
        transition: transform 0.3s ease;
    }
    
    .manual-pred-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.5);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ensure sidebar is accessible */
    [data-testid="stSidebar"] {
        position: relative !important;
    }
    
    /* Add padding to main content to account for hamburger */
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "rf_48hr_model.joblib"
FEATURE_COLS_PATH = BASE_DIR / "artifacts" / "feature_cols.json"
# Try multiple possible data paths
DATA_PATH_OPTIONS = [
    BASE_DIR / "data" / "forecast_output_next15.csv",
    BASE_DIR / "data" / "forecast_output_48hr.csv",
    BASE_DIR / "data" / "forecast_output.csv"
]
DATA_PATH = None
for path in DATA_PATH_OPTIONS:
    if path.exists():
        DATA_PATH = path
        break

# ---------------- Helper Functions ----------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_zenith_approximation(dt, latitude=28.6):
    """
    Simple zenith angle approximation based on date and time.
    latitude default is approximate for India (can be adjusted).
    """
    day_of_year = dt.timetuple().tm_yday
    hour_decimal = dt.hour + dt.minute / 60.0
    
    # Solar declination angle
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # Hour angle
    solar_noon = 12.0
    hour_angle = 15 * (hour_decimal - solar_noon)
    
    # Zenith angle approximation
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)
    
    zenith_rad = np.arccos(
        np.sin(lat_rad) * np.sin(dec_rad) +
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    zenith_deg = np.degrees(zenith_rad)
    
    # Clamp to reasonable range
    return max(0, min(180, zenith_deg))

def create_time_options():
    """Create time options with 15-minute intervals from 00:00 to 23:45"""
    times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            time_str = f"{hour:02d}:{minute:02d}"
            times.append(time_str)
    return times

def prepare_features_for_prediction(date, time_str, irradiance, temp, humidity, cloud_cover, 
                                     wind_speed=2.5, use_default_lags=True):
    """
    Prepare feature vector for model prediction.
    For lag/rolling features, we use current values as defaults if use_default_lags=True.
    """
    # Parse time
    hour, minute = map(int, time_str.split(':'))
    
    # Create datetime object
    dt = datetime.combine(date, datetime.min.time().replace(hour=hour, minute=minute))
    
    # Compute derived features
    hour_val = dt.hour
    minute_val = dt.minute
    dayofyear = dt.timetuple().tm_yday
    hour_sin = np.sin(2 * np.pi * hour_val / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_val / 24.0)
    
    # Compute zenith
    zenith = compute_zenith_approximation(dt)
    
    # For lag features, use current values if no history available
    if use_default_lags:
        irradiance_lag1 = irradiance
        irradiance_lag2 = irradiance
        irradiance_roll3 = irradiance
        irradiance_roll5 = irradiance
        temp_lag1 = temp
        temp_lag2 = temp
        temp_roll3 = temp
        temp_roll5 = temp
    else:
        # If you have historical data, compute these properly
        irradiance_lag1 = irradiance
        irradiance_lag2 = irradiance
        irradiance_roll3 = irradiance
        irradiance_roll5 = irradiance
        temp_lag1 = temp
        temp_lag2 = temp
        temp_roll3 = temp
        temp_roll5 = temp
    
    # Create feature dictionary in the order expected by the model
    features = {
        'irradiance': irradiance,
        'temp': temp,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'zenith': zenith,
        'wind_speed_10m': wind_speed,
        'hour': hour_val,
        'minute': minute_val,
        'dayofyear': dayofyear,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'irradiance_lag1': irradiance_lag1,
        'irradiance_lag2': irradiance_lag2,
        'irradiance_roll3': irradiance_roll3,
        'irradiance_roll5': irradiance_roll5,
        'temp_lag1': temp_lag1,
        'temp_lag2': temp_lag2,
        'temp_roll3': temp_roll3,
        'temp_roll5': temp_roll5
    }
    
    return features, dt

def train_and_save_model():
    """Train a Random Forest model from train.csv and save it."""
    train_path = BASE_DIR / "data" / "train.csv"
    if not train_path.exists():
        st.error("Training data not found at data/train.csv")
        return None
    try:
        with open(FEATURE_COLS_PATH, 'r') as f:
            cols = json.load(f)

        train = pd.read_csv(train_path)
        train["target_next15_kW"] = train["actual_power_kW"].shift(-1)
        train = train.dropna().reset_index(drop=True)

        X_train = train[cols]
        y_train = train["target_next15_kW"]

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, MODEL_PATH)
        return rf
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model, training it first if not present."""
    if not MODEL_PATH.exists():
        with st.spinner("Model not found — training now (this takes ~1 min on first deploy)..."):
            return train_and_save_model()
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_feature_columns():
    """Load feature column names"""
    try:
        with open(FEATURE_COLS_PATH, 'r') as f:
            feature_cols = json.load(f)
        return feature_cols
    except Exception as e:
        st.error(f"Error loading feature columns: {str(e)}")
        return None

# ---------------- Load Model and Features ----------------
model = load_model()
feature_cols = load_feature_columns()

# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌞 SolarCast</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0; font-size: 1.1rem; font-weight: 500;">Solar Power Prediction</p>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🧭 Navigation", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔮 Manual Prediction"],
    label_visibility="collapsed"
)

# Add prominent manual prediction button
if page == "📊 Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.2rem; border-radius: 12px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4); margin: 1rem 0;">
            <h3 style="color: white; margin: 0 0 0.5rem 0; font-size: 1.2rem;">🔮 Try Manual Prediction</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
                Predict power output with custom parameters
            </p>
        </div>
    """, unsafe_allow_html=True)

# Add info section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h4 style="color: white; margin: 0 0 0.5rem 0;">ℹ️ About</h4>
        <p style="color: rgba(255,255,255,0.95); font-size: 0.9rem; margin: 0; line-height: 1.5;">
            Advanced ML-powered solar power forecasting system using Random Forest regression.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------- Dashboard Page ----------------
if page == "📊 Dashboard":
    # Load historical data
    if DATA_PATH is None:
        st.error("❌ Forecast data file not found. Please ensure one of the following files exists:")
        for path in DATA_PATH_OPTIONS:
            st.write(f"  - {path}")
        st.stop()
    
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
    col_actual = "actual_next15_kW" if "actual_next15_kW" in df.columns else df.filter(like="actual").columns[0]
    col_model = "pred_model_kW" if "pred_model_kW" in df.columns else None
    col_base = "pred_baseline_kW" if "pred_baseline_kW" in df.columns else None
    
    # Sidebar controls
    st.sidebar.header("⚙ Controls")
    
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    date_range = st.sidebar.date_input(
        "Select date range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle tuple return shape
    if isinstance(date_range, (tuple, list)):
        start_date, end_date = date_range[0], date_range[1]
    else:
        start_date, end_date = date_range, date_range
    
    show_baseline = st.sidebar.checkbox("Show Baseline", True)
    show_model = st.sidebar.checkbox("Show Model", True)
    clean_mode = st.sidebar.checkbox("🧹 Clean presentation mode (Daily + smooth)", value=True)
    
    # Filter data
    mask = (df[time_col].dt.date >= start_date) & (df[time_col].dt.date <= end_date)
    dfv = df.loc[mask].copy()
    
    # Header with enhanced styling
    st.markdown('<div class="main-header">🌞 SolarCast — Solar Power Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #667eea; font-size: 1.2rem; margin-bottom: 2rem;">⚡ Advanced Solar Power Forecasting with Machine Learning</p>', unsafe_allow_html=True)
    
    # Add prominent link to manual prediction
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 12px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4); margin-bottom: 2rem;">
            <h3 style="color: white; margin: 0; font-size: 1.3rem;">🔮 Want to make a custom prediction?</h3>
            <p style="color: rgba(255,255,255,0.95); margin: 0.5rem 0 0 0; font-size: 1rem;">
                Click the <strong>☰</strong> menu button on the top-left to open navigation, then select "🔮 Manual Prediction"
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced KPIs with icons and better styling
    if col_model and col_model in dfv.columns:
        k_mae = float(np.mean(np.abs(dfv[col_actual] - dfv[col_model])))
        k_rmse = rmse(dfv[col_actual], dfv[col_model])
        k_mape = mape(dfv[col_actual], dfv[col_model])
        avg_actual = float(dfv[col_actual].mean())
        avg_predicted = float(dfv[col_model].mean())
        max_power = float(dfv[col_actual].max())
    else:
        k_mae = k_rmse = k_mape = avg_actual = avg_predicted = max_power = np.nan
    
    # First row of metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if not np.isnan(k_mae):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                <h4 style="margin: 0; opacity: 0.9;">📊 Mean Absolute Error</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{k_mae:,.2f}</h2>
                <p style="margin: 0; opacity: 0.8;">kW</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("MAE", "—")
    
    with k2:
        if not np.isnan(k_rmse):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);">
                <h4 style="margin: 0; opacity: 0.9;">📈 Root Mean Square Error</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{k_rmse:,.2f}</h2>
                <p style="margin: 0; opacity: 0.8;">kW</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("RMSE", "—")
    
    with k3:
        if not np.isnan(k_mape):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);">
                <h4 style="margin: 0; opacity: 0.9;">🎯 Mean Absolute % Error</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{k_mape:,.2f}</h2>
                <p style="margin: 0; opacity: 0.8;">%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("MAPE", "—")
    
    with k4:
        if not np.isnan(max_power):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);">
                <h4 style="margin: 0; opacity: 0.9;">⚡ Peak Power</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{max_power:,.0f}</h2>
                <p style="margin: 0; opacity: 0.8;">kW</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Peak", "—")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Main chart section
    st.markdown("### 📉 Power Output Analysis")
    
    # Build plotting frame from filtered data
    plot_cols = [time_col, col_actual]
    if show_model and col_model:
        plot_cols.append(col_model)
    if show_baseline and col_base:
        plot_cols.append(col_base)
    dfp = dfv[plot_cols].copy()
    
    if clean_mode:
        # Presentation view: daily mean + 3-day rolling average
        dfp = (dfp.set_index(time_col)
                  .resample("1D")
                  .mean()
                  .rolling(window=3, min_periods=1)
                  .mean()
                  .reset_index())
        line_w = 2.5
    else:
        line_w = 1.5
    
    # Create interactive Altair chart
    chart_data = dfp.copy()
    chart_data[time_col] = pd.to_datetime(chart_data[time_col])
    
    # Prepare data for Altair
    base = alt.Chart(chart_data).encode(
        x=alt.X(f'{time_col}:T', title='Date & Time', axis=alt.Axis(format='%Y-%m-%d %H:%M'))
    )
    
    # Create layers for each series
    layers = []
    
    # Actual line
    actual_line = base.mark_line(
        stroke='#2E86AB',
        strokeWidth=line_w,
        point=alt.OverlayMarkDef(size=30, filled=True, fill='#2E86AB')
    ).encode(
        y=alt.Y(f'{col_actual}:Q', title='Power (kW)', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(f'{time_col}:T', format='%Y-%m-%d %H:%M'), alt.Tooltip(f'{col_actual}:Q', format='.2f', title='Actual Power (kW)')]
    ).properties(
        title='Actual vs Predicted Power Output',
        width=800,
        height=400
    )
    layers.append(actual_line)
    
    # Model prediction line
    if show_model and col_model and col_model in chart_data.columns:
        model_line = base.mark_line(
            stroke='#F18F01',
            strokeWidth=line_w,
            strokeDash=[5, 5],
            point=alt.OverlayMarkDef(size=20, filled=True, fill='#F18F01')
        ).encode(
            y=alt.Y(f'{col_model}:Q', title='Power (kW)'),
            tooltip=[alt.Tooltip(f'{time_col}:T', format='%Y-%m-%d %H:%M'), alt.Tooltip(f'{col_model}:Q', format='.2f', title='Predicted Power (kW)')]
        )
        layers.append(model_line)
    
    # Baseline line
    if show_baseline and col_base and col_base in chart_data.columns:
        baseline_line = base.mark_line(
            stroke='#A23B72',
            strokeWidth=line_w,
            strokeDash=[2, 2],
            point=alt.OverlayMarkDef(size=15, filled=True, fill='#A23B72')
        ).encode(
            y=alt.Y(f'{col_base}:Q', title='Power (kW)'),
            tooltip=[alt.Tooltip(f'{time_col}:T', format='%Y-%m-%d %H:%M'), alt.Tooltip(f'{col_base}:Q', format='.2f', title='Baseline Power (kW)')]
        )
        layers.append(baseline_line)
    
    # Combine layers
    chart = alt.layer(*layers).resolve_scale(y='shared').interactive()
    st.altair_chart(chart, use_container_width=True)
    
    # Also create matplotlib version for download
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dfp[time_col], dfp[col_actual], label="Actual", linewidth=line_w, color='#2E86AB', marker='o', markersize=3)
    
    if show_baseline and col_base and col_base in dfp.columns:
        ax.plot(dfp[time_col], dfp[col_base], label="Baseline", linestyle=":", linewidth=line_w, color='#A23B72', marker='s', markersize=2)
    
    if show_model and col_model and col_model in dfp.columns:
        ax.plot(dfp[time_col], dfp[col_model], label="Predicted (Random Forest)", linestyle="--", linewidth=line_w, color='#F18F01', marker='^', markersize=2)
    
    ax.set_xlabel("Datetime", fontsize=12, fontweight='bold')
    ax.set_ylabel("Power (kW)", fontsize=12, fontweight='bold')
    ax.set_title("Power Output Forecast Comparison", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.autofmt_xdate()
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    plt.close(fig)
    
    # Optional sample view
    if st.checkbox("Show Sample View (First 100 Points)"):
        sample = dfp.head(100)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        hs = []
        hs.append(ax2.plot(sample[time_col], sample[col_actual], label="Actual", linewidth=2, color='#2E86AB')[0])
        if show_model and col_model and col_model in sample.columns:
            hs.append(ax2.plot(sample[time_col], sample[col_model], label="Predicted (Random Forest)", linestyle="--", linewidth=2, color='#F18F01')[0])
        if show_baseline and col_base and col_base in sample.columns:
            hs.append(ax2.plot(sample[time_col], sample[col_base], label="Baseline", linestyle=":", linewidth=2, color='#A23B72')[0])
        ax2.legend(handles=hs)
        ax2.grid(True, alpha=0.3)
        fig2.autofmt_xdate()
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Summary Statistics Section
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    
    if col_model and col_model in dfv.columns:
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.9); font-weight: 600; font-size: 0.9rem;">Average Actual</p>
                <h3 style="margin: 0.5rem 0 0 0; color: white; font-weight: 700; font-size: 1.5rem;">{avg_actual:,.2f} kW</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #f093fb; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.9); font-weight: 600; font-size: 0.9rem;">Average Predicted</p>
                <h3 style="margin: 0.5rem 0 0 0; color: white; font-weight: 700; font-size: 1.5rem;">{avg_predicted:,.2f} kW</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            min_actual = float(dfv[col_actual].min())
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4facfe; box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.9); font-weight: 600; font-size: 0.9rem;">Minimum Power</p>
                <h3 style="margin: 0.5rem 0 0 0; color: white; font-weight: 700; font-size: 1.5rem;">{min_actual:,.2f} kW</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            std_actual = float(dfv[col_actual].std())
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #43e97b; box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.9); font-weight: 600; font-size: 0.9rem;">Std Deviation</p>
                <h3 style="margin: 0.5rem 0 0 0; color: white; font-weight: 700; font-size: 1.5rem;">{std_actual:,.2f} kW</h3>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data table with better styling
    st.markdown("### 🧾 Forecast Data (filtered)")
    st.dataframe(
        dfv.head(300).style.background_gradient(subset=[col_actual] if col_actual in dfv.columns else [], cmap='YlOrRd'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <p style="margin: 0; color: #667eea;">
            📁 <strong>Data Source:</strong> {DATA_PATH.name} | 
            📊 <strong>Rows:</strong> {len(dfv):,} | 
            🕒 <strong>Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Manual Prediction Page ----------------
elif page == "🔮 Manual Prediction":
    st.markdown('<div class="main-header">🔮 Manual Power Prediction</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if model is None or feature_cols is None:
        st.error("⚠️ Model or feature columns could not be loaded. Please check the artifacts folder.")
        st.stop()
    
    # Create form for manual prediction
    with st.form("prediction_form"):
        st.subheader("📝 Enter Prediction Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date input with validation
            today = datetime.now().date()
            max_allowed_date = today + timedelta(days=2)
            
            selected_date = st.date_input(
                "📅 Date",
                value=today,
                min_value=None,  # Allow past dates
                max_value=max_allowed_date,
                help=f"Select any past date or up to 2 days in the future (max: {max_allowed_date})"
            )
            
            # Validate date
            if selected_date > max_allowed_date:
                st.error(f"❌ Date cannot be more than 2 days in the future. Maximum allowed: {max_allowed_date}")
                st.stop()
            
            # Time dropdown with 15-minute intervals
            time_options = create_time_options()
            selected_time = st.selectbox(
                "⏰ Time (15-minute intervals)",
                options=time_options,
                index=time_options.index("12:00") if "12:00" in time_options else 0,
                help="Select time in 15-minute intervals from 00:00 to 23:45"
            )
        
        with col2:
            st.info("💡 **Model can predict power output for the next 2 days from today.**")
            if selected_date > today:
                days_ahead = (selected_date - today).days
                st.success(f"✅ Predicting for {days_ahead} day(s) ahead")
        
        st.markdown("---")
        st.subheader("🌡️ Weather Parameters")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            irradiance = st.number_input(
                "☀️ Irradiance (W/m²)",
                min_value=0.0,
                max_value=1500.0,
                value=500.0,
                step=10.0,
                help="Solar irradiance in watts per square meter"
            )
            
            temp = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=25.0,
                step=0.5,
                help="Ambient temperature in Celsius"
            )
        
        with col4:
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
                help="Relative humidity percentage"
            )
            
            cloud_cover = st.number_input(
                "☁️ Cloud Cover (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Cloud cover percentage"
            )
        
        with col5:
            wind_speed = st.number_input(
                "💨 Wind Speed (m/s)",
                min_value=0.0,
                max_value=50.0,
                value=2.5,
                step=0.1,
                help="Wind speed at 10m height in meters per second"
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Power Output", use_container_width=True)
    
    # Process prediction
    if submitted:
        try:
            # Prepare features
            features_dict, dt = prepare_features_for_prediction(
                selected_date, selected_time, irradiance, temp, humidity, cloud_cover, wind_speed
            )
            
            # Create feature vector in the correct order
            feature_vector = np.array([[features_dict[col] for col in feature_cols]])
            
            # Make prediction - extract scalar value properly
            prediction_result = model.predict(feature_vector)
            # Convert to Python float - handle numpy arrays of any shape
            if isinstance(prediction_result, np.ndarray):
                # Convert to list first, then extract first element - this handles nested arrays
                prediction_list = prediction_result.tolist()
                # Recursively get first scalar value
                while isinstance(prediction_list, list) and len(prediction_list) > 0:
                    prediction_list = prediction_list[0]
                prediction = float(prediction_list)
            else:
                prediction = float(prediction_result)
            
            # Display results with enhanced styling
            st.markdown("---")
            st.markdown('<h2 style="text-align: center; color: #667eea; margin-bottom: 2rem;">✨ Prediction Results ✨</h2>', unsafe_allow_html=True)
            
            # Result cards with animations
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.markdown(f"""
                <div class="prediction-card" style="animation-delay: 0s;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <h3 style="margin: 0; opacity: 0.9; font-size: 1.1rem;">⚡ Predicted Power</h3>
                            <h1 style="font-size: 3.5rem; margin: 0.5rem 0; font-weight: 700;">{prediction:.2f}</h1>
                            <p style="font-size: 1.3rem; margin: 0; opacity: 0.9;">kW</p>
                        </div>
                        <div style="font-size: 4rem; opacity: 0.3;">⚡</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                efficiency_ratio = float((prediction / max(irradiance, 1)) * 100 if irradiance > 0 else 0)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 2rem; border-radius: 15px; color: white; box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
                            animation: slideInUp 0.6s ease-out; animation-delay: 0.2s;">
                    <h3 style="margin: 0; opacity: 0.9; font-size: 1.1rem;">📅 Date & Time</h3>
                    <p style="font-size: 1.4rem; margin: 0.8rem 0; font-weight: 600;"><strong>{dt.strftime('%Y-%m-%d')}</strong></p>
                    <p style="font-size: 1.4rem; margin: 0; font-weight: 600;"><strong>{selected_time}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 2rem; border-radius: 15px; color: white; box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
                            animation: slideInUp 0.6s ease-out; animation-delay: 0.4s;">
                    <h3 style="margin: 0; opacity: 0.9; font-size: 1.1rem;">🎯 Efficiency Ratio</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0; font-weight: 700;">{efficiency_ratio:.2f}</h1>
                    <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature details - Only show user-input weather parameters
            st.markdown("---")
            st.subheader("📋 Weather Parameters Used")
            
            # Only show the weather parameters that user manually inputs
            user_input_features = ['irradiance', 'temp', 'humidity', 'cloud_cover', 'wind_speed_10m']
            
            # Create display names for better readability
            display_names = {
                'irradiance': '☀️ Irradiance (W/m²)',
                'temp': '🌡️ Temperature (°C)',
                'humidity': '💧 Humidity (%)',
                'cloud_cover': '☁️ Cloud Cover (%)',
                'wind_speed_10m': '💨 Wind Speed (m/s)'
            }
            
            feature_display_data = {
                'Parameter': [display_names.get(feat, feat) for feat in user_input_features],
                'Value': [features_dict[feat] for feat in user_input_features]
            }
            
            feature_display = pd.DataFrame(feature_display_data)
            st.dataframe(feature_display, use_container_width=True, hide_index=True)
            
            # Enhanced Visualization Section
            st.markdown("---")
            st.markdown("### 📈 Interactive Visualizations")
            
            # Create two columns for visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Bar chart showing key inputs using Altair
                key_features = ['Irradiance', 'Temperature', 'Humidity', 'Cloud Cover']
                key_values = [features_dict['irradiance'], features_dict['temp'], 
                             features_dict['humidity'], features_dict['cloud_cover']]
                
                # Normalize for visualization
                normalized_values = []
                max_vals = [1500, 50, 100, 100]
                for val, max_val in zip(key_values, max_vals):
                    normalized_values.append((val / max_val) * 100)
                
                # Create Altair bar chart
                source = pd.DataFrame({
                    'Parameter': key_features,
                    'Normalized Value (%)': normalized_values,
                    'Actual Value': key_values
                })
                
                bars = alt.Chart(source).mark_bar(
                    cornerRadiusTopLeft=5,
                    cornerRadiusTopRight=5
                ).encode(
                    x=alt.X('Parameter:N', title='Parameters', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Normalized Value (%):Q', title='Normalized Value (%)', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color('Parameter:N', scale=alt.Scale(
                        domain=key_features,
                        range=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4']
                    )),
                    tooltip=[alt.Tooltip('Parameter:N'), 
                            alt.Tooltip('Actual Value:Q', format='.1f', title='Actual Value'),
                            alt.Tooltip('Normalized Value (%):Q', format='.1f', title='Normalized (%)')]
                ).properties(
                    title='Key Input Parameters',
                    width=300,
                    height=300
                )
                st.altair_chart(bars, use_container_width=True)
            
            with viz_col2:
                # Gauge-style visualization for prediction
                max_power = 1000
                power_percentage = float(min(100, (prediction / max_power) * 100))
                
                # Create a simple gauge visualization
                gauge_data = pd.DataFrame({
                    'Metric': ['Predicted Power'],
                    'Value': [power_percentage],
                    'Max': [100]
                })
                
                gauge = alt.Chart(gauge_data).mark_bar(
                    cornerRadiusTopLeft=10,
                    cornerRadiusTopRight=10
                ).encode(
                    x=alt.X('Metric:N', title=''),
                    y=alt.Y('Value:Q', title='Percentage (%)', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color('Value:Q', scale=alt.Scale(
                        domain=[0, 50, 100],
                        range=['#f5576c', '#f093fb', '#43e97b']
                    )),
                    tooltip=[alt.Tooltip('Value:Q', format='.1f', title='Power Output (%)')]
                ).properties(
                    title=f'Power Output: {prediction:.2f} kW',
                    width=300,
                    height=300
                )
                st.altair_chart(gauge, use_container_width=True)
            
            # Additional matplotlib visualization
            st.markdown("#### Detailed Parameter Analysis")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Weather Parameters & Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
            
            # 1. Bar chart
            bars = ax1.bar(key_features, normalized_values, color=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'], edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Normalized Value (%)', fontsize=11, fontweight='bold')
            ax1.set_title('Input Parameters (Normalized)', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
            for bar, val in zip(bars, key_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # 2. Pie chart for weather conditions
            weather_labels = ['Clear', 'Partly Cloudy', 'Cloudy']
            cloud_val = cloud_cover
            if cloud_val < 30:
                weather_values = [100-cloud_val, cloud_val, 0]
            elif cloud_val < 70:
                weather_values = [30, cloud_val-30, 70-cloud_val]
            else:
                weather_values = [0, 30, cloud_val]
            
            colors_pie = ['#FFD93D', '#6BCB77', '#4D96FF']
            ax2.pie(weather_values, labels=weather_labels, autopct='%1.1f%%', 
                   colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax2.set_title('Weather Conditions', fontsize=12, fontweight='bold')
            
            # 3. Prediction vs Input comparison
            comparison_data = {
                'Irradiance': irradiance / 10,  # Scale down for visualization
                'Predicted Power': prediction
            }
            ax3.bar(comparison_data.keys(), comparison_data.values(), 
                   color=['#667eea', '#f093fb'], edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax3.set_title('Irradiance vs Predicted Power', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            # 4. Efficiency metrics
            efficiency_metrics = {
                'Efficiency': efficiency_ratio,
                'Power Output': power_percentage
            }
            ax4.barh(list(efficiency_metrics.keys()), list(efficiency_metrics.values()),
                    color=['#43e97b', '#38f9d7'], edgecolor='black', linewidth=1.5)
            ax4.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
            ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            ax4.set_xlim(0, 100)
            ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Prediction gauge visualization
            st.markdown("---")
            col_gauge1, col_gauge2 = st.columns(2)
            
            with col_gauge1:
                st.markdown("### Power Output Range")
                # Simple gauge visualization
                max_power = 1000  # Adjust based on your typical max power
                power_percentage = float(min(100, (prediction / max_power) * 100))
                
                st.progress(power_percentage / 100)
                st.caption(f"{prediction:.2f} kW out of {max_power} kW ({power_percentage:.1f}%)")
            
            with col_gauge2:
                st.markdown("### Weather Conditions Summary")
                conditions = []
                if irradiance > 700:
                    conditions.append("☀️ High Solar Irradiance")
                elif irradiance > 300:
                    conditions.append("🌤️ Moderate Solar Irradiance")
                else:
                    conditions.append("☁️ Low Solar Irradiance")
                
                if cloud_cover < 30:
                    conditions.append("☀️ Clear Sky")
                elif cloud_cover < 70:
                    conditions.append("⛅ Partly Cloudy")
                else:
                    conditions.append("☁️ Cloudy")
                
                for cond in conditions:
                    st.write(cond)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)
