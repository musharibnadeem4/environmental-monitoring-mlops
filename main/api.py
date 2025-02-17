import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from typing import Dict, Any
import tensorflow as tf

class ModelDeployer:
    def __init__(self, model_path: str):
        """
        Initialize the model deployer with a pre-trained model
        
        :param model_path: Path to the saved machine learning model
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model
        
        :param input_data: Input features for prediction
        :return: Model predictions (AQI, co, no2, etc.)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Predict using the model
        predictions = self.model.predict(input_data)
        return predictions

    def classify_aqi(self, aqi: float) -> str:
        """Classify AQI into risk categories."""
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


# Flask Application
app = Flask(__name__)

# Initialize the model deployer
MODEL_PATH = 'optimized_pollution_lstm_model.joblib'
model_deployer = ModelDeployer(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for receiving input data and returning predictions
    """
    try:
        # Get JSON input
        input_data = request.get_json(force=True)
        
        # Extract features
        features = input_data['features']
        
        # Check if there are exactly 8 features for 1 time step
        if len(features) != 8:
            return jsonify({
                'error': 'Input data must contain exactly 8 features for one time step',
                'status': 'error'
            }), 400
        
        # Reshape input to match the expected shape (1, 24, 8)
        input_array = np.array(features).reshape(1, 1, 8)  # Reshape to (1, 1, 8)
        input_array = np.repeat(input_array, 24, axis=1)  # Repeat 24 times, making shape (1, 24, 8)
        
        # Debugging: Check the shape of input_array
        print(f"Input array shape: {input_array.shape}")

        # Make prediction
        prediction = model_deployer.predict(input_array)
        
        # Debugging: Check the shape of the prediction result
        print(f"Prediction result shape: {prediction.shape}")

        # Feature names
        features_names = ["aqi", "co", "no2", "o3", "pm2_5", "pm10", "temperature", "humidity"]
        
        # Generate AQI risk categories for each prediction
        risk_categories = []
        
        for i in range(len(features_names)):
            if i < len(prediction[0]):
                risk_categories.append(model_deployer.classify_aqi(prediction[0][i]))
            else:
                risk_categories.append("Unknown")

        # Prepare the response with features and their corresponding risk categories
        predictions_with_risk = []
        for i, feature_name in enumerate(features_names):
            if i < len(prediction[0]):
                # Ensure the predicted value is converted to a native Python float
                predicted_value = float(prediction[0][i])
                predictions_with_risk.append({
                    "feature": feature_name,
                    "predicted_value": predicted_value,
                    "risk_category": risk_categories[i]
                })
            else:
                predictions_with_risk.append({
                    "feature": feature_name,
                    "predicted_value": "Unknown",
                    "risk_category": "Unknown"
                })
        
        # Prepare the full response
        response: Dict[str, Any] = {
            'predictions': predictions_with_risk,
            'status': 'success'
        }
        
        return jsonify(response), 200
    
    except ValueError as ve:
        return jsonify({
            'error': str(ve),
            'status': 'error'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Unexpected error occurred during prediction',
            'details': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_deployer.model is not None
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
