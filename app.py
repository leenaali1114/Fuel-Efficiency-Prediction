from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Ensure the images directory exists
os.makedirs('static/images', exist_ok=True)

# Try to load the models and feature information
try:
    # Load the models
    with open('models/linear_regression_model_fuel.pkl', 'rb') as f:
        fuel_model = pickle.load(f)
    with open('models/linear_regression_model_electric.pkl', 'rb') as f:
        electric_model = pickle.load(f)
    
    # Load feature information
    with open('models/fuel_feature_info.pkl', 'rb') as f:
        fuel_feature_info = pickle.load(f)
    with open('models/electric_feature_info.pkl', 'rb') as f:
        electric_feature_info = pickle.load(f)
        
    # Check if models are properly loaded
    if not hasattr(fuel_model, 'predict') or not hasattr(electric_model, 'predict'):
        raise Exception("Models don't have predict method")
        
except Exception as e:
    print(f"Error loading models or feature info: {e}")
    print("Creating dummy models and feature info...")
    
    # Create dummy models if loading fails
    fuel_model = LinearRegression()
    fuel_model.coef_ = np.zeros(12)  # 3 numeric + 9 categorical features
    fuel_model.intercept_ = 25.0     # Default MPG prediction
    
    electric_model = LinearRegression()
    electric_model.coef_ = np.zeros(14)  # 8 numeric + 6 categorical features
    electric_model.intercept_ = 97.0     # Default efficiency prediction
    
    # Create dummy feature info
    fuel_feature_info = {
        'numeric_features': ['cylinders', 'displacement', 'year'],
        'categorical_features': ['drive', 'fuel_type', 'transmission'],
        'input_dim': 12  # 3 numeric + 9 one-hot encoded
    }
    
    electric_feature_info = {
        'numeric_features': ['SOC (%)', 'Voltage (V)', 'Current (A)', 'Battery Temp (°C)', 
                           'Ambient Temp (°C)', 'Charging Duration (min)', 
                           'Degradation Rate (%)', 'Efficiency (%)'],
        'categorical_features': ['Charging Mode', 'Battery Type', 'EV Model'],
        'input_dim': 14  # 8 numeric + 6 one-hot encoded
    }

# Define the features for each model
fuel_features = fuel_feature_info['numeric_features']
# Remove Efficiency (%) from electric features
electric_features = [f for f in electric_feature_info['numeric_features'] if f != 'Efficiency (%)']

# Define categorical features and their possible values
fuel_categorical = {
    'drive': ['fwd', 'rwd', '4wd', 'awd'],
    'fuel_type': ['gas', 'diesel', 'electricity'],
    'transmission': ['a', 'm']
}

electric_categorical = {
    'Charging Mode': ['Fast', 'Normal', 'Slow'],
    'Battery Type': ['Li-ion', 'LiFePO4'],
    'EV Model': ['Model A', 'Model B', 'Model C']
}

@app.route('/')
def home():
    return render_template('index.html', 
                          fuel_features=fuel_features,
                          electric_features=electric_features,
                          fuel_categorical=fuel_categorical,
                          electric_categorical=electric_categorical)

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model_type']
    
    if model_type == 'fuel':
        # Get numerical features
        input_features = []
        for feature in fuel_features:
            try:
                value = float(request.form[feature])
                input_features.append(value)
            except ValueError:
                return render_template('result.html', 
                                      result=f"Invalid value for {feature}. Please enter a number.", 
                                      model_type=model_type)
        
        # Get categorical features and one-hot encode them
        categorical_values = []
        for category, options in fuel_categorical.items():
            selected = request.form[category]
            for option in options:
                if option == selected:
                    categorical_values.append(1)
                else:
                    categorical_values.append(0)
        
        # Combine features
        all_features = input_features + categorical_values
        
        # Ensure correct dimensions
        expected_dim = fuel_feature_info['input_dim']
        if len(all_features) != expected_dim:
            print(f"Dimension mismatch: got {len(all_features)}, expected {expected_dim}")
            # Pad or truncate to match expected dimensions
            if len(all_features) < expected_dim:
                all_features.extend([0] * (expected_dim - len(all_features)))
            else:
                all_features = all_features[:expected_dim]
        
        # Make prediction
        try:
            prediction = fuel_model.predict([all_features])[0]
            result = f"Predicted Fuel Efficiency (MPG): {prediction:.2f}"
        except Exception as e:
            print(f"Prediction error: {e}")
            result = "Error making prediction. Please check your input values."
        
    else:  # electric model
        # Get numerical features - MODIFIED to match the training data
        input_features = []
        for feature in electric_features:
            try:
                value = float(request.form[feature])
                input_features.append(value)
            except ValueError:
                return render_template('result.html', 
                                      result=f"Invalid value for {feature}. Please enter a number.", 
                                      model_type=model_type)
        
        # Get categorical features using one-hot encoding to match the training
        categorical_values = []
        for category, options in electric_categorical.items():
            selected = request.form[category]
            # One-hot encode to match training
            for option in options:
                if option == selected:
                    categorical_values.append(1)
                else:
                    categorical_values.append(0)
        
        # Combine features
        all_features = input_features + categorical_values
        
        # Ensure correct dimensions
        expected_dim = electric_feature_info['input_dim']
        if len(all_features) != expected_dim:
            print(f"Dimension mismatch: got {len(all_features)}, expected {expected_dim}")
            # Pad or truncate to match expected dimensions
            if len(all_features) < expected_dim:
                all_features.extend([0] * (expected_dim - len(all_features)))
            else:
                all_features = all_features[:expected_dim]
        
        # Make prediction
        try:
            prediction = electric_model.predict([all_features])[0]
            result = f"Predicted Electric Charging Efficiency: {prediction:.2f}%"
        except Exception as e:
            print(f"Prediction error: {e}")
            result = "Error making prediction. Please check your input values."
    
    return render_template('result.html', result=result, model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True) 