from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load model and features
model = joblib.load('../models/flood_rf_model.pkl')
features = joblib.load('../models/flood_rf_features.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the flood risk based on the input parameters.

    Parameters
    ----------
    data : dict
        A dictionary containing the input parameters.

    Returns
    -------
    A JSON response containing the flood risk and its probability.
    """
    data = request.json
    # Create an input vector from the provided data
    input_vector = [data.get(feat, 0) for feat in features]
    # Predict the flood risk
    risk = int(model.predict([input_vector])[0])
    # Calculate the probability of the predicted risk
    prob = float(model.predict_proba([input_vector])[0][risk])
    # Return the flood risk and its probability as a JSON response
    return jsonify({'flood_risk': risk, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True, port=5000)