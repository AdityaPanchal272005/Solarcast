# app.py — SolarCast Dashboard with Manual Prediction
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="🌞 SolarCast — Solar Power Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
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

@st.cache_resource
def load_model():
    """Load the trained model"""
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
st.sidebar.title("🌞 SolarCast")
page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔮 Manual Prediction"])

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
    
    # Header
    st.markdown('<div class="main-header">🌞 SolarCast — Solar Power Prediction Dashboard</div>', unsafe_allow_html=True)
    st.caption("Predicting solar plant output with Random Forest Model")
    
    # KPIs
    if col_model and col_model in dfv.columns:
        k_mae = float(np.mean(np.abs(dfv[col_actual] - dfv[col_model])))
        k_rmse = rmse(dfv[col_actual], dfv[col_model])
        k_mape = mape(dfv[col_actual], dfv[col_model])
    else:
        k_mae = k_rmse = k_mape = np.nan
    
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("MAE (model)", f"{k_mae:,.2f} kW" if not np.isnan(k_mae) else "—")
    with k2:
        st.metric("RMSE (model)", f"{k_rmse:,.2f} kW" if not np.isnan(k_rmse) else "—")
    with k3:
        st.metric("MAPE %", f"{k_mape:,.2f}%" if not np.isnan(k_mape) else "—")
    
    # Main chart
    st.subheader("📉 Actual vs Predicted Power Output")
    
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
        line_w = 1.0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5))
    handles = []
    
    # Always plot Actual
    h_actual, = ax.plot(dfp[time_col], dfp[col_actual], label="Actual", linewidth=line_w, color='#2E86AB')
    handles.append(h_actual)
    
    # Conditionally add Baseline / Model
    if show_baseline and col_base and col_base in dfp.columns:
        h_base, = ax.plot(dfp[time_col], dfp[col_base], label="Baseline", linestyle=":", linewidth=line_w, color='#A23B72')
        handles.append(h_base)
    
    if show_model and col_model and col_model in dfp.columns:
        h_model, = ax.plot(dfp[time_col], dfp[col_model], label="Predicted (Random Forest)", linestyle="--", linewidth=line_w, color='#F18F01')
        handles.append(h_model)
    
    ax.set_xlabel("Datetime", fontsize=11)
    ax.set_ylabel("Power (kW)", fontsize=11)
    ax.set_title("Baseline vs Model Forecast Comparison", fontsize=13, fontweight='bold')
    ax.legend(handles=handles, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
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
    
    # Data table
    st.subheader("🧾 Forecast Data (filtered)")
    st.dataframe(dfv.head(300), use_container_width=True)
    
    st.markdown("---")
    st.caption(f"📁 Data Source: {DATA_PATH} | Rows in filter: {len(dfv)} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Result cards
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Power</h3>
                    <h1 style="font-size: 3rem; margin: 0;">{prediction:.2f}</h1>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">kW</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3>Prediction Date & Time</h3>
                    <p style="font-size: 1.3rem; margin: 0.5rem 0;"><strong>{dt.strftime('%Y-%m-%d')}</strong></p>
                    <p style="font-size: 1.3rem; margin: 0;"><strong>{selected_time}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result3:
                # Calculate some insights
                efficiency_ratio = float((prediction / max(irradiance, 1)) * 100 if irradiance > 0 else 0)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3>Efficiency Ratio</h3>
                    <h1 style="font-size: 2.5rem; margin: 0;">{efficiency_ratio:.2f}</h1>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature details
            st.markdown("---")
            st.subheader("📋 Feature Values Used")
            
            feature_display = pd.DataFrame({
                'Feature': feature_cols,
                'Value': [features_dict[col] for col in feature_cols]
            })
            st.dataframe(feature_display, use_container_width=True)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Prediction Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Bar chart showing key inputs
            key_features = ['irradiance', 'temp', 'humidity', 'cloud_cover']
            key_values = [features_dict[k] for k in key_features]
            
            # Normalize for visualization (scale to 0-100)
            normalized_values = []
            max_vals = [1500, 50, 100, 100]  # Max expected values
            for val, max_val in zip(key_values, max_vals):
                normalized_values.append((val / max_val) * 100)
            
            bars = ax.bar(key_features, normalized_values, color=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_ylabel('Normalized Value (%)', fontsize=11)
            ax.set_title('Key Input Parameters (Normalized)', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, key_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
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
