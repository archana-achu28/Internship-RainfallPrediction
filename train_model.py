# train_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib

# Load and prepare historical data (keep this for training only)
df = pd.read_excel(r'C:\Users\archa\Downloads\open-meteo-11.00N77.00E431m (2).xlsx', engine='openpyxl', index_col='time', parse_dates=True)
df = df.ffill()

# Features and target
X = df[['temperature_2m (°C)','dew_point_2m (°C)','relative_humidity_2m (%)', 'cloud_cover (%)',
        'wind_speed_10m (km/h)', 'weather_code (wmo code)', 'surface_pressure (hPa)',
        'pressure_msl (hPa)', 'visibility (m)', 'rain (mm)', 'is_day ()',
        'uv_index ()', 'shortwave_radiation (W/m²)']]
y = df['precipitation (mm)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Train and save model
model = RandomForestRegressor(n_estimators=70, random_state=80)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

# Evaluate
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("R²:", r2_score(y_test, model.predict(X_test)))
