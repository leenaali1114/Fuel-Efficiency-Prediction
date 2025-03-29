import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Train and save fuel efficiency model
def train_fuel_model():
    # Load data
    df = pd.read_csv('car_data.xls')
    
    # Print data info to check for missing values
    print("Fuel data info:")
    print(df.info())
    print("\nMissing values in fuel data:")
    print(df.isnull().sum())
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Shape after dropping NA: {df.shape}")
    
    # Prepare features and target
    X_numeric = df[['cylinders', 'displacement', 'year']]
    
    # One-hot encode categorical features
    categorical_features = ['drive', 'fuel_type', 'transmission']
    encoder = OneHotEncoder(sparse=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    
    # Combine numerical and categorical features
    X_encoded = np.hstack((X_numeric.values, encoded_cats))
    
    # Target variable
    y = df['combination_mpg']
    
    # Check for any remaining NaN or infinite values
    if np.isnan(X_encoded).any() or np.isinf(X_encoded).any():
        print("Warning: NaN or infinite values found in X_encoded. Replacing with zeros.")
        X_encoded = np.nan_to_num(X_encoded)
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("Warning: NaN or infinite values found in y. Replacing with zeros.")
        y = np.nan_to_num(y)
    
    # Train model
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    # Save model
    with open('models/linear_regression_model_fuel.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature information for later use
    feature_info = {
        'numeric_features': ['cylinders', 'displacement', 'year'],
        'categorical_features': categorical_features,
        'encoder': encoder,
        'input_dim': X_encoded.shape[1]  # Total number of features after encoding
    }
    
    with open('models/fuel_feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"Fuel efficiency model trained and saved successfully. Input dimension: {X_encoded.shape[1]}")

# Train and save electric charging model
def train_electric_model():
    # Load data
    df = pd.read_csv('ev_battery_charging_data (1).xls')
    
    # Print data info to check for missing values
    print("\nElectric data info:")
    print(df.info())
    print("\nMissing values in electric data:")
    print(df.isnull().sum())
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Shape after dropping NA: {df.shape}")
    
    # Print column names to verify
    print("Available columns:", df.columns.tolist())
    
    # Prepare features and target - MODIFIED based on notebook analysis
    # Exclude 'Efficiency (%)' if it's actually the target variable
    X_numeric = df[['SOC (%)', 'Voltage (V)', 'Current (A)', 'Battery Temp (°C)', 
            'Ambient Temp (°C)', 'Charging Duration (min)', 
            'Degradation Rate (%)']]  # Removed 'Efficiency (%)' if it's the target
    
    # One-hot encode categorical features
    categorical_features = ['Charging Mode', 'Battery Type', 'EV Model']
    encoder = OneHotEncoder(sparse=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    
    # Combine numerical and categorical features
    X_encoded = np.hstack((X_numeric.values, encoded_cats))
    
    # Target variable - MODIFIED based on notebook analysis
    y = df['Efficiency (%)']  # Changed from 'Optimal Charging Duration Class'
    
    # Check for any remaining NaN or infinite values
    if np.isnan(X_encoded).any() or np.isinf(X_encoded).any():
        print("Warning: NaN or infinite values found in X_encoded. Replacing with zeros.")
        X_encoded = np.nan_to_num(X_encoded)
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("Warning: NaN or infinite values found in y. Replacing with zeros.")
        y = np.nan_to_num(y)
    
    # Print shapes to verify
    print(f"X shape: {X_encoded.shape}, y shape: {y.shape}")
    print(f"Number of features: {X_encoded.shape[1]}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    # Save model
    with open('models/linear_regression_model_electric.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature information for later use
    feature_info = {
        'numeric_features': X_numeric.columns.tolist(),  # Use actual column names
        'categorical_features': categorical_features,
        'encoder': encoder,
        'input_dim': X_encoded.shape[1],  # Total number of features after encoding
        'target': 'Efficiency (%)'  # Store target variable name
    }
    
    with open('models/electric_feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"Electric charging model trained and saved successfully. Input dimension: {X_encoded.shape[1]}")

if __name__ == "__main__":
    try:
        train_fuel_model()
    except Exception as e:
        print(f"Error in fuel model training: {e}")
    
    try:
        train_electric_model()
    except Exception as e:
        print(f"Error in electric model training: {e}") 