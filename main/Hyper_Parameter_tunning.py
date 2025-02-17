import pandas as pd
import numpy as np
import itertools
import modelt
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

class PollutionLSTMTuner:
    def __init__(self, data_path, columns_to_convert):
        """
        Initialize the tuner with data paths and columns to convert
        
        :param data_path: Dictionary of file paths for different datasets
        :param columns_to_convert: List of columns to preprocess
        """
        self.data_path = data_path
        self.columns_to_convert = columns_to_convert
        self.scaler = None
        self.data = None

    def load_and_preprocess_data(self):
        """Load and preprocess datasets."""
        # Load datasets
        pollution = pd.read_csv(self.data_path['pollution'], parse_dates=["timestamp"], header=0)
        air_quality = pd.read_csv(self.data_path['air_quality'], parse_dates=["timestamp"], header=0)
        weather = pd.read_csv(self.data_path['weather'], parse_dates=["timestamp"], header=0)

        # Merge datasets on timestamp
        pollution.set_index("timestamp", inplace=True)
        air_quality.set_index("timestamp", inplace=True)
        weather.set_index("timestamp", inplace=True)

        data = pollution.join(air_quality, how="outer").join(weather, how="outer").reset_index()

        # Ensure numeric types and handle errors
        data[self.columns_to_convert] = data[self.columns_to_convert].apply(pd.to_numeric, errors='coerce')

        # Handle missing values
        data.fillna(method="ffill", inplace=True)
        data.fillna(method="bfill", inplace=True)

        # Normalize data using MinMaxScaler
        self.scaler = MinMaxScaler()
        data[self.columns_to_convert] = self.scaler.fit_transform(data[self.columns_to_convert])

        self.data = data
        return data

    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM input."""
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = data[i + seq_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def build_model(self, lstm_units, dropout_rate, learning_rate):
        """
        Build LSTM model with given hyperparameters
        
        :param lstm_units: Number of LSTM units
        :param dropout_rate: Dropout rate for regularization
        :param learning_rate: Learning rate for optimizer
        :return: Compiled Keras model
        """
        model = Sequential([
            LSTM(lstm_units, activation='relu', return_sequences=True, 
                 input_shape=(self.seq_length, len(self.columns_to_convert))),
            Dropout(dropout_rate),
            LSTM(lstm_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(len(self.columns_to_convert))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def run_hyperparameter_tuning(self, num_experiments=20):
        """
        Run hyperparameter tuning with multiple configurations
        
        :param num_experiments: Number of experiments to run
        :return: Best model and its configuration
        """
        # Set experiment name
        modelt.set_experiment("Pollution_LSTM_Hyperparameter_Tuning")

        # Prepare data
        data_values = self.load_and_preprocess_data()[self.columns_to_convert].values
        self.seq_length = 24  # Can be a hyperparameter too

        # Create sequences
        X, y = self.create_sequences(data_values, self.seq_length)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter search space
        hyperparameter_grid = {
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [1e-2, 1e-3, 1e-4],
            'epochs': [30, 50, 70],
            'batch_size': [16, 32, 64]
        }

        # Generate all possible combinations
        keys, values = zip(*hyperparameter_grid.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Limit to specified number of experiments
        hyperparameter_combinations = hyperparameter_combinations[:num_experiments]

        best_mae = float('inf')
        best_model = None
        best_config = None

        # Iterate through hyperparameter combinations
        for config in hyperparameter_combinations:
            with modelt.start_run():
                # Log all hyperparameters
                modelt.log_params(config)

                # Build and train model
                model = self.build_model(
                    config['lstm_units'], 
                    config['dropout_rate'], 
                    config['learning_rate']
                )

                # Train the model
                history = model.fit(
                    X_train, y_train,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    validation_data=(X_test, y_test),
                    verbose=0
                )

                # Evaluate the model
                loss, mae = model.evaluate(X_test, y_test, verbose=0)

                # Predict and inverse transform
                y_pred = model.predict(X_test)
                y_pred = self.scaler.inverse_transform(y_pred)
                y_test = self.scaler.inverse_transform(y_test)

                # Calculate AQI-specific metrics
                predicted_aqi = y_pred[:, 0]
                actual_aqi = y_test[:, 0]
                
                mse = mean_squared_error(actual_aqi, predicted_aqi)
                mae_aqi = mean_absolute_error(actual_aqi, predicted_aqi)

                # Log metrics
                modelt.log_metrics({
                    "test_loss": loss,
                    "test_mae": mae,
                    "aqi_mse": mse,
                    "aqi_mae": mae_aqi
                })

                # Track best model
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_config = config

                # Log model artifact
                modelt.keras.log_model(model, f"lstm_model_{len(hyperparameter_combinations)}")

        return best_model, best_config, best_mae

def main():
    # Data paths
    data_paths = {
        'pollution': "data/pollution_data.csv",
        'air_quality': "data/air_quality_data.csv",
        'weather': "data/forecast_data.csv"
    }

    # Columns to convert and normalize
    columns_to_convert = ["aqi", "co", "no2", "o3", "pm2_5", "pm10", "temperature", "humidity"]

    # Initialize and run hyperparameter tuning
    tuner = PollutionLSTMTuner(data_paths, columns_to_convert)
    
    best_model, best_config, best_mae = tuner.run_hyperparameter_tuning(num_experiments=20)

    print("\nBest Model Configuration:")
    for param, value in best_config.items():
        print(f"{param}: {value}")
    print(f"Best Mean Absolute Error: {best_mae}")

    # Optionally save the best model
    best_model.save("best_pollution_lstm_model.h5")
    print("Best model saved to best_pollution_lstm_model.h5")



if __name__ == "__main__":
    main()