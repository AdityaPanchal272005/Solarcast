# app.py — SolarCast Professional Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="SolarCast — Solar Output Forecast", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Site Configuration
PLANT_CAPACITY_MW = 5.0
PLANT_CAPACITY_KW = PLANT_CAPACITY_MW * 1000
EFFICIENCY = 0.20  # DC to AC conversion efficiency

# --- Robust Pathing ---
# Use Path(__file__) to get the directory of this script (E:\Solarcast)
# This ensures it finds the /data folder correctly, regardless of where you run it from.
APP_DIR = Path(__file__).parent 

# Set paths relative to the app's location
FORECAST_DATA_PATH = APP_DIR / "data" / "forecast_output_48hr.csv"
MODEL_DATA_PATH = APP_DIR / "data" / "solar_data_model.csv"

# ---------------- Helper Functions ----------------
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        model_path = APP_DIR / "artifacts" / "rf_48hr_model.joblib"
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_feature_columns():
    """Load feature columns required by the model"""
    try:
        feature_path = APP_DIR / "artifacts" / "feature_cols.json"
        with open(feature_path, 'r') as f:
            feature_cols = json.load(f)
        return feature_cols
    except Exception as e:
        st.warning(f"Could not load feature columns: {e}")
        return None

def prepare_features_for_prediction(irradiance, temp, humidity, cloud_cover, wind_speed, 
                                   datetime_input, historical_data=None):
    """Prepare features for model prediction from user input"""
    # Extract time features
    hour = datetime_input.hour
    minute = datetime_input.minute
    dayofyear = datetime_input.timetuple().tm_yday
    
    # Cyclical time features
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # Calculate zenith angle (simplified - actual would use location and date)
    # For now, approximate based on hour: zenith is lowest (sun highest) at noon
    solar_noon_hour = 12
    hour_from_noon = abs(hour - solar_noon_hour)
    zenith = min(90 + hour_from_noon * 5, 180)  # Approximate zenith
    
    # Calculate lags and rolling means if historical data available
    if historical_data is not None and len(historical_data) > 0:
        irradiance_lag1 = historical_data[-1]['irradiance'] if len(historical_data) >= 1 else irradiance
        irradiance_lag2 = historical_data[-2]['irradiance'] if len(historical_data) >= 2 else irradiance
        temp_lag1 = historical_data[-1]['temp'] if len(historical_data) >= 1 else temp
        temp_lag2 = historical_data[-2]['temp'] if len(historical_data) >= 2 else temp
        
        # Rolling means (simplified - would need more historical data)
        irradiance_roll3 = np.mean([h['irradiance'] for h in historical_data[-3:]]) if len(historical_data) >= 3 else irradiance
        irradiance_roll5 = np.mean([h['irradiance'] for h in historical_data[-5:]]) if len(historical_data) >= 5 else irradiance
        temp_roll3 = np.mean([h['temp'] for h in historical_data[-3:]]) if len(historical_data) >= 3 else temp
        temp_roll5 = np.mean([h['temp'] for h in historical_data[-5:]]) if len(historical_data) >= 5 else temp
    else:
        # Use current values as fallback
        irradiance_lag1 = irradiance_lag2 = irradiance
        temp_lag1 = temp_lag2 = temp
        irradiance_roll3 = irradiance_roll5 = irradiance
        temp_roll3 = temp_roll5 = temp
    
    # Create feature dictionary in the order expected by model
    features = {
        'irradiance': irradiance,
        'temp': temp,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'zenith': zenith,
        'wind_speed_10m': wind_speed,
        'hour': hour,
        'minute': minute,
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
    
    return features

def calculate_baseline_persistence(actual_series, prediction_length):
    """Naïve baseline: persistence model (use last known value)"""
    if len(actual_series) == 0:
        return np.zeros(prediction_length)
    last_value = actual_series[-1] if len(actual_series) > 0 else 0
    return np.full(prediction_length, last_value)

def calculate_baseline_mean(actual_series, prediction_length):
    """Naïve baseline: mean of historical values"""
    if len(actual_series) == 0:
        return np.zeros(prediction_length)
    mean_value = np.mean(actual_series)
    return np.full(prediction_length, mean_value)

def calculate_metrics(actual, predicted):
    """Calculate accuracy metrics"""
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def get_weather_insights(df_weather, start_idx, horizon):
    """Extract weather-driven insights from the forecast period"""
    weather_window = df_weather.iloc[start_idx:start_idx+horizon]
    
    insights = {}
    if 'irradiance' in weather_window.columns:
        insights['max_irradiance'] = weather_window['irradiance'].max()
        insights['min_irradiance'] = weather_window['irradiance'].min()
        insights['avg_irradiance'] = weather_window['irradiance'].mean()
        insights['cloudy_periods'] = (weather_window['cloud_cover'] > 50).sum() if 'cloud_cover' in weather_window.columns else 0
    
    if 'temp' in weather_window.columns:
        insights['max_temp'] = weather_window['temp'].max()
        insights['min_temp'] = weather_window['temp'].min()
        insights['avg_temp'] = weather_window['temp'].mean()
    
    if 'humidity' in weather_window.columns:
        insights['avg_humidity'] = weather_window['humidity'].mean()
    
    if 'wind_speed_10m' in weather_window.columns:
        insights['max_wind'] = weather_window['wind_speed_10m'].max()
        insights['avg_wind'] = weather_window['wind_speed_10m'].mean()
    
    return insights

# ---------------- Load data ----------------
@st.cache_data
def load_data():
    """Loads 48-hr forecast data and the full actuals data."""
    try:
        # Load the 48-hour predictions
        df_forecast = pd.read_csv(FORECAST_DATA_PATH, parse_dates=['datetime_start'])
        
        # Load the full actuals dataset with weather features
        df_model = pd.read_csv(MODEL_DATA_PATH, parse_dates=['datetime'])
        
        if df_forecast.empty or df_model.empty:
            st.error("Data files are empty.")
            return None, None, None
            
        return df_forecast, df_model, df_model  # Return weather data too
            
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}.")
        st.info(f"Looking for files at:\n1. {FORECAST_DATA_PATH}\n2. {MODEL_DATA_PATH}")
        st.warning("Please ensure you have run the *entire* Jupyter Notebook (including Phase 4) to generate these files.")
        return None, None, None

# ---------------- Main Header ----------------
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌞 SolarCast</h1>
    <p style='color: white; font-size: 1.2em; margin: 10px 0;'>Professional Solar Output Forecasting System</p>
    <p style='color: rgba(255,255,255,0.9); font-size: 0.9em; margin: 5px 0;'>
        Predicting 0-48h solar output for 5 MW site at 15-min intervals
    </p>
</div>
""", unsafe_allow_html=True)

# Load data and model
df_forecast, df_model, df_weather = load_data()
model = load_model()
feature_cols = load_feature_columns()

# ---------------- Main Tabs ----------------
tab1, tab2 = st.tabs(["📊 Forecast Analysis", "🔮 Manual Prediction"])

# ==================== TAB 1: FORECAST ANALYSIS ====================
with tab1:
    if df_forecast is None or df_model is None:
        st.error("❌ Unable to load forecast data. Please ensure data files are available.")
        st.stop()
    
    st.subheader("📈 Historical Forecast Analysis")
    st.markdown("Analyze past 48-hour forecasts against actual solar output.")
    
    # Professional forecast selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get list of available forecast start times
        forecast_dates = df_forecast['datetime_start'].dt.strftime('%Y-%m-%d %H:%M')
        forecast_dates_list = forecast_dates.tolist()
        
        selected_forecast = st.selectbox(
            "Select Forecast Start Time:",
            options=forecast_dates_list,
            index=0,
            help="Choose a forecast start time from the historical test set"
        )
    
    # Find the index for the selected forecast
    selected_index = df_forecast[forecast_dates == selected_forecast].index[0]

    # Get the actual datetime from the forecast dataframe (more reliable)
    start_dt = df_forecast.loc[selected_index, 'datetime_start']

    # --- Get data for the selected 48-hour window ---
    HORIZON = 192 # 48 hours * 4 steps/hr

    # 1. Get the 192 predicted values
    pred_cols = [f"pred_T+{i+1}" for i in range(HORIZON)]
    y_pred_series = df_forecast.loc[selected_index, pred_cols].values

    # 2. Get the 192 actual values
    # We find the *actual* start time in the original model dataframe
    # Try exact match first
    model_start_index_list = df_model[df_model['datetime'] == start_dt].index

    # If no exact match, try finding the closest datetime within 15 minutes
    if len(model_start_index_list) == 0:
        # Calculate time differences and find closest match
        time_diffs = (df_model['datetime'] - start_dt).abs()
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs.loc[min_diff_idx]
        
        # Only use if within 15 minutes (15 min = 15*60 seconds = 900 seconds)
        if min_diff.total_seconds() <= 900:
            model_start_index_list = [min_diff_idx]
            st.warning(f"⚠️ Exact datetime not found. Using closest match: {df_model.loc[min_diff_idx, 'datetime']} (difference: {min_diff})")
        else:
            # Try matching by string format (in case of timezone or format differences)
            start_dt_str = start_dt.strftime('%Y-%m-%d %H:%M')
            df_model_str = df_model['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            model_start_index_list = df_model[df_model_str == start_dt_str].index
            
            if len(model_start_index_list) == 0:
                st.error(f"❌ Could not find start datetime {start_dt} in {MODEL_DATA_PATH}.")
                st.info("💡 **Troubleshooting:**")
                st.info(f"- Selected forecast start: {start_dt}")
                st.info(f"- Model data range: {df_model['datetime'].min()} to {df_model['datetime'].max()}")
                st.info(f"- Available forecast times: {len(df_forecast)} forecasts")
                st.info("- Please ensure the forecast data matches the model data timestamps")
                st.stop()

    model_start_index = model_start_index_list[0]

    # The actuals are the *next* 192 steps
    y_actual_series = df_model.loc[model_start_index + 1 : model_start_index + HORIZON, 'actual_power_kW'].values
    actual_datetimes = df_model.loc[model_start_index + 1 : model_start_index + HORIZON, 'datetime'].values

    # Get historical data for baseline (use data before forecast start)
    historical_data = df_model.loc[:model_start_index, 'actual_power_kW'].values

    # Calculate naïve baselines
    baseline_persistence = calculate_baseline_persistence(historical_data, HORIZON)
    baseline_mean = calculate_baseline_mean(historical_data, HORIZON)

    # Use persistence as primary baseline (common naïve approach for time series)
    y_baseline = baseline_persistence

    # ---------------- Create Plotting DataFrame ----------------
    if len(y_pred_series) == HORIZON and len(y_actual_series) == HORIZON:
        plot_df = pd.DataFrame({
            'Datetime': actual_datetimes,
            'Actual_Power_kW': y_actual_series,
            'Predicted_Power_kW': y_pred_series,
            'Baseline_Power_kW': y_baseline
        })
        
        # Calculate metrics for model and baseline
        model_metrics = calculate_metrics(y_actual_series, y_pred_series)
        baseline_metrics = calculate_metrics(y_actual_series, y_baseline)
        
        # Improvement over baseline
        rmse_improvement = ((baseline_metrics['RMSE'] - model_metrics['RMSE']) / baseline_metrics['RMSE']) * 100
        mae_improvement = ((baseline_metrics['MAE'] - model_metrics['MAE']) / baseline_metrics['MAE']) * 100
        
        # --- KPIs Section ---
        st.subheader("📊 Forecast Accuracy Metrics")
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            st.metric("Model RMSE", f"{model_metrics['RMSE']:,.2f} kW", 
                     delta=f"{rmse_improvement:.1f}% vs baseline", delta_color="inverse")
        with k2:
            st.metric("Model MAE", f"{model_metrics['MAE']:,.2f} kW",
                     delta=f"{mae_improvement:.1f}% vs baseline", delta_color="inverse")
        with k3:
            st.metric("Model MAPE", f"{model_metrics['MAPE']:.2f}%")
        with k4:
            st.metric("Baseline RMSE", f"{baseline_metrics['RMSE']:,.2f} kW")
        
        # Baseline vs Model comparison
        st.subheader("📈 Model vs Baseline Comparison")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Baseline RMSE", f"{baseline_metrics['RMSE']:,.2f} kW")
        with b2:
            st.metric("Baseline MAE", f"{baseline_metrics['MAE']:,.2f} kW")
        with b3:
            st.metric("Baseline MAPE", f"{baseline_metrics['MAPE']:.2f}%")
        
        # Melt for Altair plot
        plot_df_melted = plot_df.melt('Datetime', var_name='Type', value_name='Power (kW)')

        # ---------------- Main Chart (Altair) ----------------
        st.subheader(f"📉 48-Hour Forecast vs. Actual (Starting: {selected_forecast})")

        # Create the line chart with baseline
        line_chart = alt.Chart(plot_df_melted).mark_line(interpolate='step-after', point=False, strokeWidth=2).encode(
            x=alt.X('Datetime:T', title='Date and Time'),
            y=alt.Y('Power (kW):Q', title='Power (kW)', scale=alt.Scale(domain=[0, PLANT_CAPACITY_KW * 1.1])),
            color=alt.Color('Type:N', 
                            title='Legend', 
                            scale=alt.Scale(
                                domain=['Actual_Power_kW', 'Predicted_Power_kW', 'Baseline_Power_kW'], 
                                range=['#1f77b4', '#ff7f0e', '#d62728'],
                                type='ordinal'
                            )),
            strokeDash=alt.StrokeDash('Type:N',
                                        scale=alt.Scale(
                                            domain=['Actual_Power_kW', 'Predicted_Power_kW', 'Baseline_Power_kW'],
                                            range=[[0], [0], [5, 5]]
                                        )),
            tooltip=[
                alt.Tooltip('Datetime:T', format='%Y-%m-%d %H:%M'), 
                'Type:N', 
                alt.Tooltip('Power (kW):Q', format=',.2f')
            ]
        ).properties(
            title=f"48-Hour Forecast Comparison ({PLANT_CAPACITY_MW} MW Site)",
            width=800,
            height=400
        ).interactive() # Make it zoomable/pannable

        st.altair_chart(line_chart, use_container_width=True)
        
        # ---------------- Weather-Driven Insights ----------------
        st.subheader("🌤️ Weather-Driven Insights")
    
        # Get weather data for the forecast period
        weather_insights = get_weather_insights(df_weather, model_start_index + 1, HORIZON)
        
        # Display weather metrics
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            if 'avg_irradiance' in weather_insights:
                st.metric("Avg Irradiance", f"{weather_insights['avg_irradiance']:.1f} W/m²",
                         delta=f"Max: {weather_insights['max_irradiance']:.1f}")
        with w2:
            if 'avg_temp' in weather_insights:
                st.metric("Avg Temperature", f"{weather_insights['avg_temp']:.1f} °C",
                         delta=f"Range: {weather_insights['min_temp']:.1f}-{weather_insights['max_temp']:.1f}°C")
        with w3:
            if 'avg_humidity' in weather_insights:
                st.metric("Avg Humidity", f"{weather_insights['avg_humidity']:.1f}%")
        with w4:
            if 'cloudy_periods' in weather_insights:
                total_periods = HORIZON
                cloudy_pct = (weather_insights['cloudy_periods'] / total_periods) * 100
                st.metric("Cloudy Periods", f"{weather_insights['cloudy_periods']}/{total_periods}",
                         delta=f"{cloudy_pct:.1f}%")
        
        # Weather correlation analysis
        if model_start_index + 1 + HORIZON <= len(df_weather):
            weather_df = df_weather.iloc[model_start_index + 1:model_start_index + HORIZON + 1].copy()
            weather_df['Predicted_Power'] = y_pred_series
            weather_df['Actual_Power'] = y_actual_series
            
            # Calculate correlations
            insights_text = []
            if 'irradiance' in weather_df.columns:
                irr_corr = weather_df['irradiance'].corr(weather_df['Actual_Power'])
                insights_text.append(f"**Irradiance-Power Correlation**: {irr_corr:.3f} (strongly {'positive' if irr_corr > 0.7 else 'moderate' if irr_corr > 0.3 else 'weak'})")
            
            if 'cloud_cover' in weather_df.columns:
                cloud_corr = weather_df['cloud_cover'].corr(weather_df['Actual_Power'])
                insights_text.append(f"**Cloud Cover Impact**: {abs(cloud_corr):.3f} correlation (negative correlation expected)")
            
            if insights_text:
                st.info(" | ".join(insights_text))
            
            # Weather vs Power visualization
            if 'irradiance' in weather_df.columns and len(weather_df) == HORIZON:
                st.subheader("📊 Weather Impact on Solar Output")
                
                # Prepare data for visualization
                weather_viz_df = pd.DataFrame({
                    'datetime': actual_datetimes,
                    'Actual_Power': y_actual_series,
                    'irradiance': weather_df['irradiance'].values if 'irradiance' in weather_df.columns else np.zeros(HORIZON)
                })
                
                # Create dual-axis chart for irradiance and power
                base = alt.Chart(weather_viz_df).encode(
                    x=alt.X('datetime:T', title='Date and Time')
                )
                
                power_chart = base.mark_line(color='#ff7f0e', strokeWidth=2).encode(
                    y=alt.Y('Actual_Power:Q', title='Power (kW)', axis=alt.Axis(titleColor='#ff7f0e'))
                )
                
                irradiance_chart = base.mark_line(color='#2ca02c', strokeWidth=2, strokeDash=[5, 5]).encode(
                    y=alt.Y('irradiance:Q', title='Irradiance (W/m²)', axis=alt.Axis(titleColor='#2ca02c', orient='right'))
                )
                
                weather_chart = alt.layer(power_chart, irradiance_chart).resolve_scale(
                    y='independent'
                ).properties(
                    title="Solar Power vs Irradiance",
                    width=800,
                    height=300
                ).interactive()
                
                st.altair_chart(weather_chart, use_container_width=True)

        # ---------------- Error Analysis ----------------
        st.subheader("📉 Error Analysis")
        
        # Calculate errors
        plot_df['Model_Error'] = plot_df['Actual_Power_kW'] - plot_df['Predicted_Power_kW']
        plot_df['Baseline_Error'] = plot_df['Actual_Power_kW'] - plot_df['Baseline_Power_kW']
        
        # Error distribution chart
        error_df = pd.DataFrame({
            'Datetime': actual_datetimes,
            'Model Error (kW)': plot_df['Model_Error'],
            'Baseline Error (kW)': plot_df['Baseline_Error']
        })
        error_df_melted = error_df.melt('Datetime', var_name='Type', value_name='Error (kW)')
        
        error_chart = alt.Chart(error_df_melted).mark_line(point=True).encode(
            x=alt.X('Datetime:T', title='Date and Time'),
            y=alt.Y('Error (kW):Q', title='Error (kW)', 
                    scale=alt.Scale(domain=[error_df_melted['Error (kW)'].min() * 1.1, 
                                           error_df_melted['Error (kW)'].max() * 1.1])),
            color=alt.Color('Type:N', 
                            scale=alt.Scale(domain=['Model Error (kW)', 'Baseline Error (kW)'],
                                           range=['#ff7f0e', '#d62728'])),
            tooltip=[
                alt.Tooltip('Datetime:T', format='%Y-%m-%d %H:%M'),
                'Type:N',
                alt.Tooltip('Error (kW):Q', format=',.2f')
            ]
        ).properties(
            title="Forecast Errors Over Time",
            width=800,
            height=300
        ).interactive()
        
        st.altair_chart(error_chart, use_container_width=True)
        
        # Error statistics
        err1, err2 = st.columns(2)
        with err1:
            st.write("**Model Error Statistics:**")
            st.write(f"- Mean Error: {plot_df['Model_Error'].mean():.2f} kW")
            st.write(f"- Std Dev: {plot_df['Model_Error'].std():.2f} kW")
            st.write(f"- Max Over-prediction: {plot_df['Model_Error'].min():.2f} kW")
            st.write(f"- Max Under-prediction: {plot_df['Model_Error'].max():.2f} kW")
        
        with err2:
            st.write("**Baseline Error Statistics:**")
            st.write(f"- Mean Error: {plot_df['Baseline_Error'].mean():.2f} kW")
            st.write(f"- Std Dev: {plot_df['Baseline_Error'].std():.2f} kW")
            st.write(f"- Max Over-prediction: {plot_df['Baseline_Error'].min():.2f} kW")
            st.write(f"- Max Under-prediction: {plot_df['Baseline_Error'].max():.2f} kW")
        
        # ---------------- Data table ----------------
        with st.expander("🔍 View Raw Forecast Data for this Window"):
            st.dataframe(plot_df[['Datetime', 'Actual_Power_kW', 'Predicted_Power_kW', 'Baseline_Power_kW']])
        
        # ---------------- Summary Report ----------------
        st.markdown("---")
        st.subheader("📋 Summary Report")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Model Performance:**")
            st.write(f"- ✅ RMSE: {model_metrics['RMSE']:.2f} kW ({(1-model_metrics['MAPE']/100)*100:.1f}% accuracy)")
            st.write(f"- ✅ MAE: {model_metrics['MAE']:.2f} kW")
            st.write(f"- ✅ MAPE: {model_metrics['MAPE']:.2f}%")
            st.write(f"- ✅ Improvement over baseline: {rmse_improvement:.1f}% RMSE reduction")
        
        with summary_col2:
            st.write("**Forecast Characteristics:**")
            st.write(f"- 📅 Forecast Horizon: 48 hours")
            st.write(f"- ⏱️ Time Resolution: 15 minutes ({HORIZON} steps)")
            st.write(f"- 🏭 Site Capacity: {PLANT_CAPACITY_MW} MW")
            st.write(f"- 📊 Baseline Method: Persistence (last known value)")

    else:
        st.error(f"Data integrity error: Could not align predicted ({len(y_pred_series)}) and actual ({len(y_actual_series)}) data. Please check data files.")

# ==================== TAB 2: MANUAL PREDICTION ====================
with tab2:
    st.subheader("🔮 Manual Solar Power Prediction")
    st.markdown("Enter weather parameters to predict solar output power using the trained Random Forest model.")
    
    if model is None:
        st.error("❌ Unable to load prediction model. Please ensure the model file is available.")
        st.info("Expected model path: `artifacts/rf_48hr_model.joblib`")
        st.stop()
    
    if feature_cols is None:
        st.warning("⚠️ Feature columns not loaded. Using default feature order.")
        feature_cols = ["irradiance", "temp", "humidity", "cloud_cover", "zenith", "wind_speed_10m",
                       "hour", "minute", "dayofyear", "hour_sin", "hour_cos",
                       "irradiance_lag1", "irradiance_lag2", "irradiance_roll3", "irradiance_roll5",
                       "temp_lag1", "temp_lag2", "temp_roll3", "temp_roll5"]
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("### Weather Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date and time input
            pred_date = st.date_input("Date", value=datetime.now().date())
            pred_time = st.time_input("Time", value=datetime.now().time())
            
            # Weather parameters
            irradiance = st.number_input(
                "Solar Irradiance (W/m²)",
                min_value=0.0,
                max_value=1500.0,
                value=500.0,
                step=10.0,
                help="Direct solar irradiance in watts per square meter"
            )
            
            temp = st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=25.0,
                step=1.0,
                help="Ambient air temperature in Celsius"
            )
            
            humidity = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=5.0,
                help="Relative humidity percentage"
            )
        
        with col2:
            cloud_cover = st.number_input(
                "Cloud Cover (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                help="Cloud cover percentage (0 = clear, 100 = overcast)"
            )
            
            wind_speed = st.number_input(
                "Wind Speed (m/s)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Wind speed at 10m height in meters per second"
            )
            
            st.markdown("---")
            st.markdown("### Site Information")
            st.info(f"**Site Capacity:** {PLANT_CAPACITY_MW} MW ({PLANT_CAPACITY_KW} kW)")
            st.info(f"**Efficiency:** {EFFICIENCY*100:.0f}%")
        
        submit_button = st.form_submit_button("🚀 Predict Solar Output", use_container_width=True)
    
    if submit_button:
        # Combine date and time
        pred_datetime = datetime.combine(pred_date, pred_time)
        
        # Prepare features
        features_dict = prepare_features_for_prediction(
            irradiance=irradiance,
            temp=temp,
            humidity=humidity,
            cloud_cover=cloud_cover,
            wind_speed=wind_speed,
            datetime_input=pred_datetime
        )
        
        # Convert to DataFrame in correct feature order
        # Ensure all features are present and in the correct order
        feature_df = pd.DataFrame([features_dict])
        
        # Reorder columns to match model expectations
        if feature_cols is not None:
            # Ensure all required features exist
            missing_features = set(feature_cols) - set(feature_df.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                st.stop()
            feature_df = feature_df[feature_cols]
        else:
            # Use default order if feature_cols not available
            feature_df = feature_df[[
                'irradiance', 'temp', 'humidity', 'cloud_cover', 'zenith', 'wind_speed_10m',
                'hour', 'minute', 'dayofyear', 'hour_sin', 'hour_cos',
                'irradiance_lag1', 'irradiance_lag2', 'irradiance_roll3', 'irradiance_roll5',
                'temp_lag1', 'temp_lag2', 'temp_roll3', 'temp_roll5'
            ]]
        
        # Make prediction
        try:
            # Get prediction (may be numpy array, ensure we get scalar)
            prediction_result = model.predict(feature_df)
            
            # Extract scalar value - handle both array and scalar returns
            # Use .flat or ravel() to get flattened view, then .item() to get Python scalar
            if isinstance(prediction_result, np.ndarray):
                # Flatten to 1D and get first element as Python scalar
                predicted_power = float(np.ravel(prediction_result)[0])
            elif isinstance(prediction_result, (list, tuple)):
                # For lists/tuples, get first element
                first_val = prediction_result[0]
                if isinstance(first_val, np.ndarray):
                    # If nested array, flatten and get first element
                    predicted_power = float(np.ravel(first_val)[0])
                else:
                    predicted_power = float(first_val)
            elif hasattr(prediction_result, 'item'):
                # Has item method (like numpy scalar)
                predicted_power = float(prediction_result.item())
            else:
                # Try direct conversion
                try:
                    predicted_power = float(prediction_result)
                except (TypeError, ValueError):
                    # If that fails, try converting to array first
                    predicted_power = float(np.ravel(prediction_result)[0])
            
            # Ensure prediction is within bounds (all values are now scalars)
            predicted_power = max(0.0, min(predicted_power, float(PLANT_CAPACITY_KW)))
            
            # Calculate percentage of capacity
            capacity_percentage = (predicted_power / PLANT_CAPACITY_KW) * 100
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Result cards
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.metric(
                    "Predicted Power Output",
                    f"{predicted_power:.2f} kW",
                    delta=f"{predicted_power/1000:.2f} MW"
                )
            
            with r2:
                st.metric(
                    "Capacity Utilization",
                    f"{capacity_percentage:.1f}%",
                    delta=f"{PLANT_CAPACITY_KW - predicted_power:.2f} kW remaining"
                )
            
            with r3:
                # Calculate energy for 15-min interval
                energy_15min = (predicted_power * 0.25) / 1000  # kWh for 15 minutes
                st.metric(
                    "Energy (15 min)",
                    f"{energy_15min:.3f} kWh",
                    delta=f"{energy_15min * 4:.3f} kWh/hour"
                )
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Prediction Visualization")
            
            # Gauge chart (simplified)
            fig_data = pd.DataFrame({
                'Type': ['Predicted Output', 'Remaining Capacity'],
                'Power (kW)': [predicted_power, max(0, PLANT_CAPACITY_KW - predicted_power)]
            })
            
            chart = alt.Chart(fig_data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("Power (kW):Q", stack=True),
                color=alt.Color("Type:N", 
                    scale=alt.Scale(domain=['Predicted Output', 'Remaining Capacity'],
                                   range=['#ff7f0e', '#e0e0e0'])),
                tooltip=['Type:N', alt.Tooltip('Power (kW):Q', format='.2f')]
            ).properties(
                width=400,
                height=400,
                title="Capacity Utilization"
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Input summary
            with st.expander("📋 Input Parameters Summary"):
                summary_df = pd.DataFrame({
                    'Parameter': ['Date & Time', 'Irradiance', 'Temperature', 'Humidity', 
                                'Cloud Cover', 'Wind Speed'],
                    'Value': [
                        pred_datetime.strftime('%Y-%m-%d %H:%M'),
                        f"{irradiance:.1f} W/m²",
                        f"{temp:.1f} °C",
                        f"{humidity:.1f}%",
                        f"{cloud_cover:.1f}%",
                        f"{wind_speed:.1f} m/s"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Prediction Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                if irradiance < 200:
                    st.warning("⚠️ **Low Irradiance**: Cloudy conditions detected. Power output may be reduced.")
                elif irradiance > 800:
                    st.success("✅ **High Irradiance**: Optimal sunny conditions. Maximum power generation expected.")
                
                if cloud_cover > 70:
                    st.warning("⚠️ **High Cloud Cover**: Significant reduction in solar output expected.")
            
            with insight_col2:
                if temp > 35:
                    st.info("ℹ️ **High Temperature**: Panel efficiency may be reduced at very high temperatures.")
                
                if capacity_percentage > 90:
                    st.success("🎯 **Near Capacity**: Plant operating at peak performance!")
                elif capacity_percentage < 20:
                    st.info("ℹ️ **Low Utilization**: Conditions not optimal for solar generation.")
        
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption(f"📁 Data Source: {FORECAST_DATA_PATH} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("💡 **Naïve Baseline**: Uses persistence model (last known value) for comparison")

