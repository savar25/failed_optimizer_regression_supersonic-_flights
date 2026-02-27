import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def smooth_profile(profile, window_size=5):
    return uniform_filter1d(profile, size=window_size)

def load_csv_files(folder_path):
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

def prepare_data_for_cnn(df):
    X = df[['time']].values.reshape(-1, 1, 1)
    y = df['h'].values.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_cnn(filters=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
    Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def plot_input_cases(df):
    plt.figure(figsize=(10, 5))
    for case_id in df['id'].unique():
        case_df = df[df['id'] == case_id]
        plt.plot(case_df['time'], case_df['h'], alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Altitude (h)")
    plt.title("All Input Cases")
    plt.show()

def plot_general_prediction(df, model):
    case_id = df['id'].unique()[0]
    time_values = df[df['id'] == case_id]['time'].values
    
    # Prepare the feature set for time-based predictions
    X_case = df[df['id'] == case_id][['time']]
    #time_values = df[['time']].values.reshape(-1, 1, 1)
    predicted_h = model.predict(X_case).flatten()
    smoothed_h = savgol_filter(predicted_h, window_length=11, polyorder=3)
    plt.figure(figsize=(12, 6))
    plt.plot(time_values, predicted_h, label="Predicted Profile", linestyle='dotted', color='blue')
    plt.plot(time_values, smoothed_h, label="Smoothed Profile", color='red')
    plt.xlabel("Time")
    plt.ylabel("Altitude (h)")
    plt.title("Generalized h vs Time Profile (CNN Regression)")
    plt.legend()
    plt.show()

def plot_hyperparameter_effects(df, param_name, values):
    case_id = df['id'].unique()[0]
    time_values = df[df['id'] == case_id]['time'].values
    
    # Prepare the feature set for time-based predictions
    X_case = df[df['id'] == case_id][['time']]
    #time_values = df[['time']].values.reshape(-1, 1, 1)
    plt.figure(figsize=(10, 5))
    for value in values:
        model = build_cnn(**{param_name: value})
        model.fit(X_train, y_train, epochs=10, verbose=0)
        predicted_h = model.predict(X_case).flatten()
        plt.plot(time_values, predicted_h, label=f"{param_name}={value}")
    plt.xlabel("Time")
    plt.ylabel("Altitude (h)")
    plt.title(f"Effect of {param_name} on Predictions")
    plt.legend()
    plt.show()

# Main Execution
folder_path = "data/csv_files"
df = load_csv_files(folder_path)
X_train, X_test, y_train, y_test = prepare_data_for_cnn(df)
cnn_model = build_cnn()
cnn_model.fit(X_train, y_train, epochs=32, verbose=1)
plot_general_prediction(df, cnn_model)
plot_input_cases(df)
plot_hyperparameter_effects(df, 'filters', [16, 32, 64])
plot_hyperparameter_effects(df, 'learning_rate', [0.01, 0.001, 0.0001])
plot_hyperparameter_effects(df, 'dropout', [0.1, 0.2, 0.5])
