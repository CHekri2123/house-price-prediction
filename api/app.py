from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load all the best models.
models = {
    "Linear Regression": joblib.load("../models/lasso_regression_best.pkl"),
    "Ridge Regression": joblib.load("../models/ridge_regression_best.pkl"),
    "Lasso Regression": joblib.load("../models/lasso_regression_best.pkl"),
    "Decision Tree": joblib.load("../models/decision_tree_best.pkl"),
    "Random Forest": joblib.load("../models/random_forest_best.pkl"),
}

# Define all the features.
FEATURES = [
    "longitude", "latitude", "housing_median_age", "rooms_per_household",
    "bedrooms_per_room", "log_median_income", "log_population_per_household"
]
SCALING_FEATURES = ["longitude", "latitude", "housing_median_age"]

# Load the StandardScaler (fit from training data)
scaler = joblib.load("../models/scaler.pkl")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure all required features are provided
        if not all(feature in data for feature in FEATURES):
            return jsonify({"error": "Missing required features"}), 400

        # Convert input data to NumPy array
        input_data = np.array([[data[feature] for feature in FEATURES]])

        # Apply Standard Scaling only to required features
        input_data[:, :3] = scaler.transform(input_data[:, :3])  
        
        # Generate predictions from each model
        predictions = {model_name: model.predict(input_data)[0] for model_name, model in models.items()}

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
