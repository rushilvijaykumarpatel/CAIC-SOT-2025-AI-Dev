from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained Linear Regression model
model = joblib.load('lr_like_predictor.pkl')

@app.route('/')
def home():
    return "Welcome to the Likes Prediction API. Please use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from JSON input
    # Make sure these keys match your feature names exactly
    features = np.array([
        data['word_count'],
        data['char_count'],
        data['sentiment'],
        data['has_media'],
        data['hour'],
        data['company_encoded'],
        data['username_encoded'],
        data['day_encoded'],
        data['has_url'],
        data['has_hashtag'],
        data['has_mention'],
    ]).reshape(1, -1)


    # Predict log(likes + 1)
    log_pred = model.predict(features)[0]

    # Convert back to original likes scale: exp(log_pred) - 1
    pred_likes = np.expm1(log_pred)

    # Return the prediction rounded and converted to int
    return jsonify({'predicted_likes': int(round(pred_likes))})

if __name__ == '__main__':
    app.run(debug=True)
