import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure directories exist
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Training script started.")

# Load processed data
df = pd.read_csv("data/processed_data.csv")

# Define features & target
selected_features = [
    "longitude", "latitude", "housing_median_age", "rooms_per_household",
    "bedrooms_per_room", "log_median_income", "log_population_per_household"
]
target = "median_house_value"

X = df[selected_features]
y = df[target]

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with hyperparameter variations
hyperparams = {
    "Linear Regression": [LinearRegression()],
    "Ridge Regression": [Ridge(alpha=a) for a in [0.1, 0.3, 0.6, 1.0, 10.0]],
    "Lasso Regression": [Lasso(alpha=a) for a in [0.1, 0.3, 0.5, 0.7, 1.0]],
    "Decision Tree": [DecisionTreeRegressor(max_depth=d) for d in [5, 10, 15, 20]],
    "Random Forest": [RandomForestRegressor(n_estimators=100, random_state=42)]
}

best_models = {}
results = {}

for model_name, model_variations in hyperparams.items():
    best_model = None
    best_r2 = float('-inf')
    best_metrics = {}

    for model in model_variations:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:  # Keep the best model based on R²
            best_r2 = r2
            best_model = model
            best_metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R² Score": r2}

    # Save the best model of this type
    best_models[model_name] = best_model
    results[model_name] = best_metrics
    joblib.dump(best_model, f"{model_dir}/{model_name.replace(' ', '_').lower()}_best.pkl")

    print(f"{model_name} (Best) - MSE: {best_metrics['MSE']:.2f}, RMSE: {best_metrics['RMSE']:.2f}, MAE: {best_metrics['MAE']:.2f}, R²: {best_metrics['R² Score']:.4f}")
    logging.info(f"{model_name} (Best) - {best_metrics}")

print("All best models saved successfully.")
logging.info("Training script completed successfully.")
