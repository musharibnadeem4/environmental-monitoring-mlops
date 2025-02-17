import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

class OptimizedPollutionLSTM:
    def __init__(self, data_paths, columns_to_convert):
        """
        Initialize the optimized LSTM model for pollution prediction
        
        :param data_paths: Dictionary of file paths for different datasets
        :param columns_to_convert: List of columns to preprocess
        """
        self.data_paths = data_paths
        self.columns_to_convert = columns_to_convert
        self.scaler = None
        self.model = None

    def load_and_preprocess_data(self):
        """Load and preprocess datasets."""
        # Load datasets
        pollution = pd.read_csv(self.data_paths['pollution'], parse_dates=["timestamp"], header=0)
        air_quality = pd.read_csv(self.data_paths['air_quality'], parse_dates=["timestamp"], header=0)
        weather = pd.read_csv(self.data_paths['weather'], parse_dates=["timestamp"], header=0)

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

    def build_and_train_model(self):
        """
        Build and train the LSTM model with optimized hyperparameters
        
        :return: Trained model and training history
        """
        # Optimized hyperparameters
        LSTM_UNITS = 16
        DROPOUT_RATE = 0.1
        LEARNING_RATE = 0.01
        EPOCHS = 10
        BATCH_SIZE = 32
        SEQ_LENGTH = 24

        # Start MLflow run
        with mlflow.start_run():
            # Log optimized hyperparameters
            mlflow.log_params({
                "lstm_units": LSTM_UNITS,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQ_LENGTH
            })

            # Prepare data
            data = self.load_and_preprocess_data()
            data_values = data[self.columns_to_convert].values

            # Create sequences
            X, y = self.create_sequences(data_values, SEQ_LENGTH)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Build model
            model = Sequential([
                LSTM(LSTM_UNITS, activation='relu', return_sequences=True, 
                     input_shape=(SEQ_LENGTH, len(self.columns_to_convert))),
                Dropout(DROPOUT_RATE),
                LSTM(LSTM_UNITS, activation='relu'),
                Dropout(DROPOUT_RATE),
                Dense(len(self.columns_to_convert))
            ])

            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
                verbose=1
            )

            # Evaluate model
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
            mlflow.log_metrics({
                "test_loss": loss,
                "test_mae": mae,
                "aqi_mse": mse,
                "aqi_mae": mae_aqi
            })

            # Log model artifact
            mlflow.keras.log_model(model, "optimized_pollution_lstm")

            # Save model in .h5 format
            model.save("optimized_pollution_lstm_model.h5")

            # Save model using joblib
            joblib.dump(model, "optimized_pollution_lstm_model.joblib")

            # Generate predictions DataFrame
            predictions_df = self.generate_predictions(model, X_test, y_pred)

            return model, history, predictions_df

    def generate_predictions(self, model, X_test, y_pred):
        """
        Generate predictions DataFrame with timestamps and risk categories
        
        :param model: Trained LSTM model
        :param X_test: Test input sequences
        :param y_pred: Predicted values
        :return: DataFrame with predictions
        """
        def classify_aqi(aqi):
            """Classify AQI risk levels."""
            if aqi < 50:
                return "Good"
            elif 50 <= aqi < 100:
                return "Fair"
            elif 100 <= aqi < 150:
                return "Moderate"
            elif 150 <= aqi < 200:
                return "Poor"
            else:
                return "Very Poor"

        # Classify predicted AQI levels
        predicted_aqi = y_pred[:, 0]
        risk_categories = [classify_aqi(aqi) for aqi in predicted_aqi]

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            "Timestamp": pd.date_range(start="2024-12-15", periods=len(predicted_aqi), freq="h"),
            "Predicted_AQI": predicted_aqi,
            "Risk_Category": risk_categories
        })

        # Save predictions to CSV
        predictions_df.to_csv("optimized_predicted_aqi_trends.csv", index=False)

        return predictions_df

def main():
    # Set MLflow experiment
    mlflow.set_experiment("Optimized_Pollution_LSTM_Prediction")

    # Data paths
    data_paths = {
        'pollution': "data/pollution_data.csv",
        'air_quality': "data/air_quality_data.csv",
        'weather': "data/forecast_data.csv"
    }

    # Columns to convert and normalize
    columns_to_convert = ["aqi", "co", "no2", "o3", "pm2_5", "pm10", "temperature", "humidity"]

    # Initialize and train the optimized model
    pollution_lstm = OptimizedPollutionLSTM(data_paths, columns_to_convert)
    
    # Build and train the model
    model, history, predictions = pollution_lstm.build_and_train_model()

    print("\nModel Training Completed:")
    print("Model saved as 'optimized_pollution_lstm_model.h5' and 'optimized_pollution_lstm_model.joblib'")
    print("Predictions saved as 'optimized_predicted_aqi_trends.csv'")
    print("\nYou can view detailed experiment tracking by running 'mlflow ui'")

if __name__ == "__main__":
    main()
