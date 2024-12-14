
from flask import Flask, request, render_template
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os 

app = Flask(__name__)

# Load the pre-trained model
script_dir = os.path.dirname(os.path.realpath(_file_)) #get the path of current file (i.e., flask_app.py)
model_path = os.path.join(script_dir, 'xenocanto.h5') #w concatenate with model file name; nees absolute file path
# model = joblib.load(model_path)

# model_path = 'xenocanto.h5'

model = load_model(model_path)
# script_dir = os.path.dirname(os.path.realpath(_file_)) #get the path of current file (i.e., flask_app.py)
scaler_path = os.path.join(script_dir, 'scaler.pkl') #w concatenate with model file name; nees absolute file path
scaler = joblib.load(scaler_path)  

def extract_features(audio, sr):
    try:
        # Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        gfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        gfcc_mean = np.mean(gfcc.T, axis=0)
        features = np.hstack([mfcc, chroma, spectral_contrast, gfcc_mean])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        audio_file = request.files.get("audio_file")

        if audio_file:
            try:
                filepath = "temp_audio.wav"
                audio_file.save(filepath)
                
                audio, sr = librosa.load(filepath, sr=None, mono=True, res_type='kaiser_fast')
                features = extract_features(audio, sr)
                if features is None:
                    return "Error extracting features", 500
                
                features = features.reshape(1, -1)
    

                features = scaler.transform(features)
                prediction = model.predict(features)
                print(f"Model prediction: {prediction}")
                
                species = map_prediction_to_species(prediction)
                return species
            except Exception as e:
                print(f"Error during prediction: {e}")
                return "Error processing file", 500
        return "No file uploaded", 400
    return render_template("upload.html")

def map_prediction_to_species(prediction):
    try:
        prediction = tf.nn.softmax(prediction).numpy()
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        species_mapping = {
            0: 'affinis',
            1: 'asiaticus',
            2: 'horsfieldii', 
            3: 'indicus',
            4: 'mystery', 
            5: 'nipalensis',
            6: 'scolopaceus', 
            7: 'striata',
            8: 'sutorius', 
            9: 'vagabunda'
        }
        return species_mapping.get(predicted_index, "Unknown species")
    except Exception as e:
        print(f"Error mapping prediction to species: {e}")
        return "Unknown species"

if __name__ == "__main__":
    app.run(debug=True)
