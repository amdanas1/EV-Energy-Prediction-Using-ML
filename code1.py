import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("PHASE 1: INITIATION")
try:
    df = pd.read_csv('EV_Energy_Consumption_Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'EV_Energy_Consumption_Dataset.csv' not found.")
    exit()
df.columns = df.columns.str.replace('%', 'Percent', regex=False).str.replace('.', '', regex=False)
df = df.dropna()
df['Efficiency_Wh_km'] = (df['Energy_Consumption_kWh'] * 1000) / df['Distance_Travelled_km']
print("Applying Statistical Outlier Removal (5% - 95%)...")
initial_count = len(df)
lower_bound = df['Efficiency_Wh_km'].quantile(0.05)
upper_bound = df['Efficiency_Wh_km'].quantile(0.95)
df_phase1 = df[(df['Efficiency_Wh_km'] >= lower_bound) & (df['Efficiency_Wh_km'] <= upper_bound)]
print(f"Data Reduced from {initial_count} to {len(df_phase1)} samples.")
print(f"Efficiency Range Kept: {lower_bound:.2f} Wh/km to {upper_bound:.2f} Wh/km")
features = [
    'Speed_kmh',
    'Acceleration_ms2',
    'Battery_State_Percent',
    'Battery_Voltage_V',
    'Battery_Temperature_C',
    'Driving_Mode',
    'Road_Type',
    'Traffic_Condition',
    'Slope_Percent',
    'Weather_Condition',
    'Temperature_C',
    'Humidity_Percent',
    'Wind_Speed_ms',
    'Tire_Pressure_psi',
    'Vehicle_Weight_kg'
    ]
X = df_phase1[features]
y = df_phase1['Efficiency_Wh_km']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
def print_metrics(model_name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {model_name} Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f} Wh/km")
    print(f"RMSE:     {rmse:.2f} Wh/km")
print_metrics("Linear Regression (Baseline)", y_test, y_pred_lr)
print_metrics("Random Forest (Advanced)", y_test, y_pred_rf)
plt.figure(figsize=(12, 10))
sns.heatmap(df_phase1[features + ['Efficiency_Wh_km']].corr(), cmap='coolwarm', annot=False)
plt.title('Phase 1: Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('phase1_heatmap.png')
print("\nSaved: 'phase1_heatmap.png'")
plt.figure(figsize=(8, 6))
sns.histplot(df_phase1['Efficiency_Wh_km'], bins=30, kde=True, color='blue')
plt.title('Distribution of Energy Efficiency (Post-Cleaning)')
plt.xlabel('Energy Consumption (Wh/km)')
plt.ylabel('Frequency')
plt.savefig('phase1_distribution.png')
print("Saved: 'phase1_distribution.png'")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Wh/km')
plt.ylabel('Predicted Wh/km')
plt.title('Phase 1 Model Performance: Actual vs Predicted')
plt.savefig('phase1_performance.png')