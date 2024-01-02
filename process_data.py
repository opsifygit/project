import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('dummy_sensor_data.csv')

# Preprocessing
# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# Encode categorical variables
label_encoder_machine = LabelEncoder()
label_encoder_sensor = LabelEncoder()
df['Machine_ID'] = label_encoder_machine.fit_transform(df['Machine_ID'])
df['Sensor_ID'] = label_encoder_sensor.fit_transform(df['Sensor_ID'])

# Feature Engineering
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day

# save the new data
df.to_csv('processed_data.csv', index=False)

# save the label encoders
joblib.dump(label_encoder_machine, 'app/label_encoder_machine.joblib')
joblib.dump(label_encoder_sensor, 'app/label_encoder_sensor.joblib')