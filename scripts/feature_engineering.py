import numpy as np

def add_features(df):
    """Creates new features to improve model performance."""
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Ensure no negative values before log transformation
    df["median_income"] = df["median_income"].clip(lower=1e-9)
    df["population_per_household"] = df["population_per_household"].clip(lower=1e-9)

    # Apply log transformation
    df["log_median_income"] = np.log1p(df["median_income"])
    df["log_population_per_household"] = np.log1p(df["population_per_household"])

    return df
