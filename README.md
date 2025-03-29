# Efficiency Predictor

A machine learning web application that predicts both fuel efficiency for conventional vehicles and charging efficiency for electric vehicles.

## Project Overview

This application uses linear regression models to make predictions based on vehicle specifications and battery parameters. It features:

- Dual prediction capabilities (fuel efficiency and electric charging efficiency)
- Interactive web interface built with Flask
- Responsive design that works on desktop and mobile devices
- Real-time predictions based on user inputs

## Features

### Fuel Efficiency Prediction
Predicts miles per gallon (MPG) based on:
- Engine specifications (cylinders, displacement)
- Vehicle characteristics (year, drive type)
- Fuel type and transmission

### Electric Charging Efficiency Prediction
Predicts charging efficiency percentage based on:
- Battery parameters (state of charge, voltage, current)
- Temperature conditions (battery and ambient)
- Charging characteristics (duration, mode)
- Battery type and EV model

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/efficiency-predictor.git
   cd efficiency-predictor
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Ensure you have the data files:
   - `car_data.xls` - Dataset for fuel efficiency
   - `ev_battery_charging_data (1).xls` - Dataset for electric charging efficiency

## Usage

1. Train the models (only needed once):
   ```
   python train_and_save_models.py
   ```

2. Start the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

4. Select the prediction type (fuel or electric) and enter the required parameters.

5. Click "Predict Efficiency" to get your prediction.

## Project Structure

```

## Model Information

### Fuel Efficiency Model
- **Algorithm**: Linear Regression
- **Features**: 
  - Numerical: cylinders, displacement, year
  - Categorical: drive, fuel_type, transmission
- **Target**: combination_mpg (miles per gallon)

### Electric Charging Efficiency Model
- **Algorithm**: Linear Regression
- **Features**:
  - Numerical: SOC (%), Voltage (V), Current (A), Battery Temp (°C), Ambient Temp (°C), Charging Duration (min), Degradation Rate (%)
  - Categorical: Charging Mode, Battery Type, EV Model
- **Target**: Efficiency (%)

## Requirements

- Python 3.6+
- Flask
- NumPy
- Pandas
- scikit-learn
- Other dependencies listed in requirements.txt

## Future Improvements

- Add more advanced models (Random Forest, XGBoost)
- Implement feature importance visualization
- Add user authentication for saving predictions
- Expand to include more vehicle and battery types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources: [List your data sources here]
- Contributors: [List contributors here]
