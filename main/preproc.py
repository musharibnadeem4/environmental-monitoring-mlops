from sklearn.preprocessing import MinMaxScaler
import modelt
import joblib
import pandas as pd

class DataPreprocessor:
    def __init__(self, filepath):
        self.raw_data = pd.read_csv(filepath, parse_dates=['timestamp'])
        self.processed_data = None

    def handle_missing_values(self):
        missing_values = self.raw_data.isnull().sum()
        if missing_values.sum() == 0:
            self.processed_data = self.raw_data
            return
        self.processed_data = self.raw_data.fillna(self.raw_data.median())

    def remove_outliers(self, columns=None):
        if columns is None:
            columns = ['aqi', 'co', 'no2', 'o3', 'pm2_5', 'pm10']
        df = self.processed_data.copy()
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        self.processed_data = df

    def scale_features(self):
        """
        Scale numeric features using MinMaxScaler.
        """
        numeric_columns = ['aqi', 'co', 'no2', 'o3', 'pm2_5', 'pm10']
        scaler = MinMaxScaler()
        self.processed_data[numeric_columns] = scaler.fit_transform(
            self.processed_data[numeric_columns]
        )
        # Save the scaler using joblib for reproducibility
        joblib.dump(scaler, "scaler.pkl")
        modelt.log_artifact("scaler.pkl", artifact_path="preprocessing")
        return scaler
def main():
    # MLflow tracking
    modelt.set_experiment("pollution_prediction")
    
    with modelt.start_run():
        # Preprocessing pipeline
        preprocessor = DataPreprocessor('data/pollution_data_*.csv')
        preprocessor.handle_missing_values()
        preprocessor.remove_outliers()
        scaler = preprocessor.scale_features()
        
        # Prepare time series data
        X, y = preprocessor.prepare_time_series()
        
        # Log preprocessing metrics
        modelt.log_metric("data_points", len(X))
        modelt.log_metric("features", X.shape[1])
