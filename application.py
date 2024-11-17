from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

application = Flask(__name__)
app=application

# Load the pre-trained model
model = load_model('ipl_score_predictor.h5')

# Load the encoders (you can save and load these too if needed)
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

scaler = MinMaxScaler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Get the data sent from the client
    # Extract data from the request (example fields, adjust according to your input fields)
    venue = data['venue']
    batting_team = data['bat_team']
    bowling_team = data['bowl_team']
    striker = data['batsman']
    bowler = data['bowler']

    # Perform encoding on categorical data
    encoded_venue = venue_encoder.transform([venue])
    encoded_batting_team = batting_team_encoder.transform([batting_team])
    encoded_bowling_team = bowling_team_encoder.transform([bowling_team])
    encoded_striker = striker_encoder.transform([striker])
    encoded_bowler = bowler_encoder.transform([bowler])

    # Create input array for prediction
    input_data = np.array([encoded_venue, encoded_batting_team, encoded_bowling_team, encoded_striker, encoded_bowler])
    input_data = input_data.reshape(1, -1)  # Reshape to match model input shape
    input_data = scaler.transform(input_data)  # Scale input data

    # Predict score using the trained model
    predicted_score = model.predict(input_data)
    predicted_score = int(predicted_score[0, 0])

    # Return the prediction as JSON
    return jsonify({'predicted_score': predicted_score})

if __name__ == '__main__':
    app.run(host="0.0.0.0")