from flask import Flask, request, jsonify
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Class mapping
class_mapping = {
    1: 'Fire',
    2: 'Rain',
    3: 'Thunderstorm',
    4: 'WaterDrops',
    5: 'Wind',
    6: 'Silence',
    7: 'TreeFalling',
    8: 'Helicopter',
    9: 'VehicleEngine',
    10: 'Axe',
    11: 'Chainsaw',
    12: 'Generator',
    13: 'Handsaw',
    14: 'Firework',
    15: 'Gunshot',
    16: 'WoodChop',
    17: 'Whistling',
    18: 'Speaking',
    19: 'Footsteps',
    20: 'Clapping',
    21: 'Insect',
    22: 'Frog',
    23: 'BirdChirping',
    24: 'WingFlaping',
    25: 'Lion',
    26: 'WolfHowl',
    27: 'Squirrel'
}

# Load your pre-trained model
model = load_model('Models/Trained_Model.keras')

# Define folders for uploading and processing
AUDIO_UPLOAD_FOLDER = 'uploads/audio'
SPECTROGRAM_FOLDER = 'uploads/spectrograms'
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# Helper function to create a spectrogram from an audio file
def create_spectrogram(audio_file, image_file):
    fig = plt.figure(figsize=(2.24, 2.24))  # Size to match model's input (224x224)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
    fig.savefig(image_file)
    plt.close(fig)

# Helper function to preprocess the spectrogram image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict_from_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio_file = request.files['file']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if audio_file and audio_file.filename.endswith('.wav'):
        audio_path = os.path.join(AUDIO_UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        # Convert audio to spectrogram image
        spectrogram_path = os.path.join(SPECTROGRAM_FOLDER, f"{os.path.splitext(audio_file.filename)[0]}.png")
        create_spectrogram(audio_path, spectrogram_path)

        # Preprocess the image and make a prediction
        img_array = preprocess_image(spectrogram_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        predicted_class_name = class_mapping.get(predicted_class+1, 'Unknown')

        # return jsonify({
        #     'message': 'Prediction successful',
        #     'predicted_class': int(predicted_class)
        # }), 200
        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': predicted_class_name
        }), 200

    return jsonify({'error': 'Invalid file type. Only .wav files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
