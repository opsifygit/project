from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# model = joblib.load('model.pkl')
# Load the pre-trained model
model = joblib.load("model.pkl")

# Create a label encoder for categorical variables
label_encoder_sensor = joblib.load("label_encoder_sensor.joblib")
label_encoder_machine = joblib.load("label_encoder_machine.joblib")

def predict_using_model(data):
    try:
        # Get data from the request
        data = request.get_json()

        # Convert the incoming data to a DataFrame
        df = pd.DataFrame([data])

        # Encode categorical variables
        df['Machine_ID'] = label_encoder_machine.transform(df['Machine_ID'])
        df['Sensor_ID'] = label_encoder_sensor.transform(df['Sensor_ID'])

        # Feature Engineering (if needed)
        df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        df['Day'] = pd.to_datetime(df['Timestamp']).dt.day

        # Make predictions
        prediction = model.predict(df.drop(['Timestamp'], axis=1))

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.get_json())
    return predict_using_model(
        request.get_json()
    )

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')