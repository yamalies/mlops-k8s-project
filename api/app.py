# api/app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_model(model_path):
    try:
        logger.info(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Use relative path or current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
logger.info(f"Current directory: {current_dir}")
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 500
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)