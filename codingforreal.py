import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("--- PHASE 2: INITIATION ---")
try:
    df = pd.read_csv('EV_Energy_Consumption_Dataset.csv')
    print("Dataset Loaded.")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()
df.columns = df.columns.str.replace('%', 'Percent', regex=False).str.replace('.', '', regex=False)
df = df.dropna()

float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].round(3)
print("Data Preprocessing: All values rounded to 3 decimal places.")

df['Efficiency_Wh_km'] = (df['Energy_Consumption_kWh'] * 1000) / df['Distance_Travelled_km']
print("Applying Physics-Guided Filtering...")
initial_count = len(df)
# Remove sensor errors and non-driving energy spikes
df_phase2 = df[(df['Efficiency_Wh_km'] >= 100) & (df['Efficiency_Wh_km'] <= 400)]
# Remove parking/idling scenarios where Wh/km is undefined
df_phase2 = df_phase2[df_phase2['Speed_kmh'] > 20]
print(f"Dataset Refined: {initial_count} -> {len(df_phase2)} valid samples.")
df_phase2['Drag_Factor'] = df_phase2['Speed_kmh'] ** 2
df_phase2['Inertial_Force'] = df_phase2['Vehicle_Weight_kg'] * df_phase2['Acceleration_ms2']
features = [
    'Speed_kmh', 'Drag_Factor', 'Inertial_Force',
    'Acceleration_ms2', 'Battery_State_Percent', 'Battery_Voltage_V',
    'Battery_Temperature_C', 'Driving_Mode', 'Road_Type',
    'Traffic_Condition', 'Slope_Percent', 'Weather_Condition',
    'Temperature_C', 'Humidity_Percent', 'Wind_Speed_ms',
    'Tire_Pressure_psi', 'Vehicle_Weight_kg'
]
X = df_phase2[features]
y = df_phase2['Efficiency_Wh_km']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
def print_results(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {name} Results ---")
    print(f"R² Score: {r2:.4f} (Positive = Success)")
    print(f"MAE:      {mae:.2f} Wh/km")
    print(f"RMSE:     {rmse:.2f} Wh/km")
    return r2, mae
print_results("Linear Regression", y_test, y_pred_lr)
print_results("Random Forest", y_test, y_pred_rf)
# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='Predictions')
plt.plot([100, 400], [100, 400], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Efficiency (Wh/km)')
plt.ylabel('Predicted Efficiency (Wh/km)')
plt.title('Phase 2 Final: Actual vs Predicted (Random Forest)')
plt.legend()
plt.tight_layout()
plt.savefig('phase2_final_vs.png')
print("\nSaved: 'phase2_final_vs.png'")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10] # Top 10 Features
plt.figure(figsize=(10, 6))
plt.title("Phase 2: Top 10 Drivers of Energy Consumption")
plt.bar(range(10), importances[indices], align="center", color='green')
plt.xticks(range(10), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('phase2_feature_importance.png')
print("Saved: 'phase2_feature_importance.png'")