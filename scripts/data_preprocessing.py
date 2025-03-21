import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from feature_engineering import add_features  # Import feature engineering functions

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """Fills missing values in numerical columns using median."""
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    return df

def encode_categorical(df):
    """Performs One-Hot Encoding on categorical features."""
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    return df

def scale_features(df, num_features):
    """Applies Standard Scaling (Z-score) to numerical features and saves the scaler."""
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    # Save the scaler for later use in the prediction API
    joblib.dump(scaler, "models/scaler.pkl")
    
    return df

def preprocess_data(filepath):
    """Complete data preprocessing pipeline."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    
    # Apply feature engineering
    df = add_features(df)

    # Define numerical features to scale
    num_features = ["longitude", "latitude", "housing_median_age"]
    
    df = scale_features(df, num_features)
    
    return df

if __name__ == "__main__":
    df = preprocess_data("data/housing.csv")
    df.to_csv("data/processed_data.csv", index=False)
    print("Data preprocessing completed. Processed data saved.")
