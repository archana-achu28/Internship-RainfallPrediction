# app.py
import streamlit as st
import requests
import pandas as pd
import joblib
import re
import datetime
import pytz
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

# Load model
model = joblib.load("model.pkl")

# Weather code interpretation
def get_weather_summary(code):
    code_map = {
        0: "☀ Clear", 1: "🌤 Mainly Clear", 2: "⛅ Partly Cloudy", 3: "☁ Cloudy",
        45: "🌫 Fog", 48: "🌫 Rime Fog", 51: "🌦 Light Drizzle", 53: "🌦 Moderate Drizzle",
        55: "🌧 Heavy Drizzle", 61: "🌧 Light Rain", 63: "🌧 Moderate Rain",
        65: "🌧 Heavy Rain", 80: "🌦 Rain Showers", 95: "⛈ Thunderstorm"
    }
    return code_map.get(code, "🌈 Weather Unavailable")

def meters_to_km(m):
    return f"{m / 1000:.1f} km"

def get_coordinates(location_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}"
    response = requests.get(url)
    results = response.json().get('results')
    if results:
        return results[0]['latitude'], results[0]['longitude']
    return None, None

def fetch_weather_data(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,cloud_cover,"
        f"wind_speed_10m,weather_code,surface_pressure,pressure_msl,visibility,"
        f"rain,is_day,uv_index,shortwave_radiation"
        f"&daily=sunrise,sunset"
        f"&forecast_days=1&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()
    hourly = data['hourly']
    last = -1

    features = {
        'temperature_2m (°C)': hourly['temperature_2m'][last],
        'dew_point_2m (°C)': hourly['dew_point_2m'][last],
        'relative_humidity_2m (%)': hourly['relative_humidity_2m'][last],
        'cloud_cover (%)': hourly['cloud_cover'][last],
        'wind_speed_10m (km/h)': hourly['wind_speed_10m'][last],
        'weather_code (wmo code)': hourly['weather_code'][last],
        'surface_pressure (hPa)': hourly['surface_pressure'][last],
        'pressure_msl (hPa)': hourly['pressure_msl'][last],
        'visibility (m)': hourly['visibility'][last],
        'rain (mm)': hourly['rain'][last],
        'is_day ()': hourly['is_day'][last],
        'uv_index ()': hourly['uv_index'][last],
        'shortwave_radiation (W/m²)': hourly['shortwave_radiation'][last],
    }

    hourly_df = pd.DataFrame(hourly)
    hourly_df['time'] = pd.to_datetime(hourly_df['time'])

    sunrise = data["daily"]["sunrise"][0]
    sunset = data["daily"]["sunset"][0]

    return pd.DataFrame([features]), hourly_df, sunrise, sunset

def format_time(iso_time):
    return pd.to_datetime(iso_time).strftime("%I:%M %p")

# Set auto-refresh every 5 minutes
st.set_page_config(page_title="Rainfall Prediction", page_icon="🌦", layout="wide")
st.experimental_rerun_interval = 300  # Refresh every 300 seconds

st.markdown("<h1 style='text-align: center;'>🌦 Live Rainfall Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

option = st.radio("Enter location by:", ("City Name", "Coordinates", "Map"))

if option == "City Name":
    location = st.text_input("Enter city (e.g., Bengaluru)")
    if location and not re.match(r'^[A-Za-z\s]+$', location.strip()):
        st.error("⚠ Please enter a valid city name.")
        st.stop()
    if st.button("Get Prediction"):
        lat, lon = get_coordinates(location)
        if lat is None:
            st.error("Invalid city name.")
        else:
            df, hourly_df, sunrise, sunset = fetch_weather_data(lat, lon)

elif option == "Coordinates":
    lat = st.number_input("Latitude", format="%.4f")
    lon = st.number_input("Longitude", format="%.4f")
    if st.button("Get Prediction"):
        df, hourly_df, sunrise, sunset = fetch_weather_data(lat, lon)

elif option == "Map":
    st.markdown("**Select a location on the map below:**")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
    map_data = st_folium(m, width=700, height=400)
    if map_data and map_data['last_clicked'] is not None:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.success(f"Selected location: ({lat:.4f}, {lon:.4f})")
        if st.button("Get Prediction"):
            df, hourly_df, sunrise, sunset = fetch_weather_data(lat, lon)
    else:
        st.info("Click on the map to select a location.")

if "df" in locals():
    prediction = model.predict(df)[0]
    temp = df.iloc[0]['temperature_2m (°C)']
    humidity = df.iloc[0]['relative_humidity_2m (%)']
    wind = df.iloc[0]['wind_speed_10m (km/h)']
    pressure = df.iloc[0]['pressure_msl (hPa)']
    dew_point = df.iloc[0]['dew_point_2m (°C)']
    visibility = meters_to_km(df.iloc[0]['visibility (m)'])
    weather_code = int(df.iloc[0]['weather_code (wmo code)'])
    condition = get_weather_summary(weather_code)
    rain = df.iloc[0]['rain (mm)']

    st.markdown(f"### 📍 Weather at ({lat:.2f}, {lon:.2f})")
    st.markdown(f"#### 🌡 {temp:.1f}°C | {condition}")
    st.caption(f"Feels like: {temp:.1f}°C")
    st.divider()

    if prediction > 0:
        st.warning(f"🌧 Rain Expected. Predicted: {prediction:.2f} mm")
    else:
        st.success(f"☀ No rain expected. Predicted: {prediction:.2f} mm")

    if rain > 10:
        st.error("🚨 Ongoing: Heavy Rain")
    elif rain > 0:
        st.warning("🌧 Light Rain Ongoing")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("💧 Humidity", f"{humidity:.0f}%")
    col2.metric("💨 Wind", f"{wind:.1f} km/h")
    col3.metric("🌫 Visibility", visibility)

    col4, col5, col6 = st.columns(3)
    col4.metric("🌡 Dew Point", f"{dew_point:.1f}°C")
    col5.metric("📊 Pressure", f"{pressure:.0f} mb")
    col6.metric("📅 Sunrise/Sunset", f"{format_time(sunrise)} / {format_time(sunset)}")

    # Hourly Rain Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_df["time"], y=hourly_df["rain"], mode='lines+markers', name="Rain (mm)"))
    fig.update_layout(title="Hourly Rain Forecast", xaxis_title="Time", yaxis_title="Rainfall (mm)", height=400)
    st.plotly_chart(fig, use_container_width=True)
