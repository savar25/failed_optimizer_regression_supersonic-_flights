
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

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

def extract_tsfresh_features(df):
    """Extracts relevant time series features for altitude using tsfresh."""
    # df_feat, y = make_forecasting_frame(df['h'], kind='altitude', max_timeshift=10, rolling_direction=1)
    extracted_features = extract_features(df, column_id='id', column_sort='time',column_value='h',n_jobs=0)
    return extracted_features.dropna(axis=1, how='any')  # Drop NaNs to avoid issues

def prepare_train_test_data(features, test_size=0.2, val_size=0.1):
    """Splits data into training, validation, and test sets."""
    X_train, X_test = train_test_split(features, test_size=test_size, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test

def build_cnn_model(input_shape):
    """Builds a CNN model for altitude prediction."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def plot_cases(df):
    """Plots all input cases in a single figure."""
    plt.figure(figsize=(10, 5))
    for case_id, case_df in df.groupby('id'):
        plt.plot(case_df['time'], case_df['h'], label=f'Case {case_id}', alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Altitude")
    plt.title("All Input Cases")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.show()

def plot_predictions(X_test, df, model):
    """Plots model predictions against time."""
    predictions = model.predict(X_test)
    time_values = df[df['id'] == df['id'].unique()[0]]['time'].values[:len(predictions)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, predictions, label="Predicted Altitude", color='red')
    plt.xlabel("Time")
    plt.ylabel("Altitude")
    plt.title("CNN Model Predictions vs Time")
    plt.legend()
    plt.show()

# Main Execution
folder_path = "data/csv_files"  # Update this path as needed
df = load_csv_files(folder_path)
#features = extract_tsfresh_features(df)
with open('./extracted_features.csv', 'r') as f:
    extracted_features = pd.read_csv(f)
X_train, X_val, X_test = prepare_train_test_data(extracted_features)

# Reshape for CNN input
X_train = np.expand_dims(X_train.values, axis=-1)
X_val = np.expand_dims(X_val.values, axis=-1)
X_test = np.expand_dims(X_test.values, axis=-1)

# Build and train model
cnn_model = build_cnn_model(input_shape=(X_train.shape[1], 1))
cnn_model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=32, batch_size=8)

# Plot cases and predictions
plot_cases(df)
plot_predictions(extracted_features, df, cnn_model)
