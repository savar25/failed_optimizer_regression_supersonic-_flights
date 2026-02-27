import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

def smooth_profile(profile, window_size=5):
    """Applies a smoothing function to the profile to reduce noise."""
    return uniform_filter1d(profile, size=window_size)

def load_csv_files(folder_path):
    """Loads multiple CSV files with specific naming pattern and concatenates them into a single DataFrame."""
    num_cases = 50
    all_data = []
    
    for i in itertools.chain(range(1, num_cases+1)):
        fufile = os.path.join(os.path.dirname(__file__), f"Data_Files/data_case_{i}.csv")
        
        with open(fufile, 'r') as f:
            tf = pd.read_csv(f)
            for idx, series in tf.iterrows():
                time, h, v, h_sim, v_sim = series
                all_data.append([i, time, h, h_sim])
    
    tf_data = pd.DataFrame(all_data, columns=['id', 'time', 'h', 'h_sim'])
    print (tf_data)
    return tf_data


def prepare_data_for_regression(df):
    """Prepares the data for regression (using time as the feature and h as the target)."""
    # For simplicity, we can use the raw time and altitude as the features for regression.
    X = df[['time']]  # Only time is the feature
    y = df['h']  # Altitude (h) is the target
    return X, y

def train_regression_model(X, y):
    """Train a Random Forest Regressor model."""
    model = RandomForestRegressor(n_estimators=200, random_state=42,max_depth=30)
    model.fit(X, y)
    return model

def plot_hyperparameter_effects(df):
    """Plots how changes in hyperparameters affect predictions for each hyperparameter."""

    case_id = df['id'].unique()[0]
    time_values = df[df['id'] == case_id]['time'].values
    
    # Prepare the feature set for time-based predictions
    X_case = df[df['id'] == case_id][['time']]
    # Hyperparameter values to explore
    n_estimators_values = [50, 100, 200]
    max_depth_values = [10, 20, 30]
    max_features_values = ['sqrt', 'log2', 0.5]  # Correct max_features values
    
    X = df[['time']]  # Using time as the feature
    y = df['h']  # Altitude as the target variable

    # 1. Plot for different `n_estimators` (keep `max_depth` and `max_features` constant)
    plt.figure(figsize=(10, 5))
    for n_estimators in n_estimators_values:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=20, max_features='sqrt', random_state=42)
        model.fit(X, y)
        predicted_h = model.predict(X_case)
        actual_h = df[df['id'] == case_id]['h'].values
        #smoothed_predicted_h = savgol_filter(actual_h, window_length=11, polyorder=3)
        smoothed_predicted_h = smooth_profile(predicted_h, window_size=5)
        plt.plot(time_values, smoothed_predicted_h, label=f'n_estimators={n_estimators}', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Predicted Altitude (h)")
    plt.title("Effect of n_estimators on Predictions")
    plt.legend()
    plt.show()

    # 2. Plot for different `max_depth` (keep `n_estimators` and `max_features` constant)
    plt.figure(figsize=(10, 5))
    for max_depth in max_depth_values:
        model = RandomForestRegressor(n_estimators=100, max_depth=max_depth, max_features='sqrt', random_state=42)
        model.fit(X, y)
        predicted_h = model.predict(X_case)
        actual_h = df[df['id'] == case_id]['h'].values
        #smoothed_predicted_h = savgol_filter(actual_h, window_length=11, polyorder=3)
        smoothed_predicted_h = smooth_profile(predicted_h, window_size=5)
        plt.plot(time_values, smoothed_predicted_h, label=f'max_depth={max_depth}', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Predicted Altitude (h)")
    plt.title("Effect of max_depth on Predictions")
    plt.legend()
    plt.show()

    # 3. Plot for different `max_features` (keep `n_estimators` and `max_depth` constant)
    plt.figure(figsize=(10, 5))
    for max_features in max_features_values:
        model = RandomForestRegressor(n_estimators=100, max_depth=20, max_features=max_features, random_state=42)
        model.fit(X, y)
        predicted_h = model.predict(X_case)
        actual_h = df[df['id'] == case_id]['h'].values
        #smoothed_h = savgol_filter(actual_h, window_length=11, polyorder=3)
        smoothed_predicted_h = smooth_profile(predicted_h, window_size=5)
        plt.plot(time_values, smoothed_predicted_h, label=f'max_features={max_features}', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Predicted Altitude (h)")
    plt.title("Effect of max_features on Predictions")
    plt.legend()
    plt.show()


def plot_input_cases(df):
    """Plots all input cases in a single figure."""
    plt.figure(figsize=(10, 5))
    for case_id, case_df in df.groupby('id'):
        plt.plot(case_df['time'], case_df['h'], label=f'Case {case_id}', alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Altitude (h)")
    plt.title("All Input Cases")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.show()


def plot_regression_profile(df, model):
    """Generates and plots the regression-based profile using time as the feature."""
    case_id = df['id'].unique()[0]
    time_values = df[df['id'] == case_id]['time'].values
    
    # Prepare the feature set for time-based predictions
    X_case = df[df['id'] == case_id][['time']]
    
    # Predict h using the trained model for each time step
    predicted_h = model.predict(X_case)

    # Smooth the actual h values to clear noise (using Savitzky-Golay filter)
    actual_h = df[df['id'] == case_id]['h'].values
    smoothed_h = savgol_filter(predicted_h, window_length=11, polyorder=3)

    # Check the length of predicted_h and time_values for consistency
    if len(predicted_h) != len(time_values) or len(smoothed_h) != len(time_values):
        raise ValueError(f"Predicted values length ({len(predicted_h)}) or smoothed values length ({len(smoothed_h)}) "
                         f"does not match time values length ({len(time_values)}).")
    
    # Plot the result
    plt.figure(figsize=(12, 6))
    plt.plot(time_values, predicted_h, label="Predicted Profile (Regression with Time Feature)", color='blue',linestyle='dotted', linewidth=2)
    plt.plot(time_values, smoothed_h, label="Noise Cleared Profile (Smoothed)", color='red', linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Altitude (h)")
    plt.title("Generalized h vs Time Profile (Regression-based with Time as Feature)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# Main Execution
folder_path = "data/csv_files"  # Update this path as needed
df = load_csv_files(folder_path)

# Prepare the data for regression (using 'time' as feature and 'h' as the target)
X, y = prepare_data_for_regression(df)

# Train the regression model
model = train_regression_model(X, y)

# Plot the generalized profile based on the trained regression model
plot_regression_profile(df, model)

 # Plot all input cases
plot_input_cases(df)
    
    # Plot effects of hyperparameter tuning
plot_hyperparameter_effects(df)