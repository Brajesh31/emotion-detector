from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import librosa
import joblib
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)  # Enable CORS for frontend requests

# Load trained model and label encoder
MODEL_PATH = r'D:/PROJECTS/Emotion_detection_project/backend/model/cnn_audio_classifier.h5'
LABEL_ENCODER_PATH = r'D:/PROJECTS/Emotion_detection_project/backend/model/label_encoder.pkl'

try:
    model = load_model(MODEL_PATH)
    
    # Handle sklearn version compatibility issue
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    logger.info("Model and label encoder loaded successfully")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Label encoder classes: {label_encoder.classes_}")
    
    # Verify sklearn compatibility
    try:
        # Test the label encoder with a dummy prediction
        test_labels = np.array([0])
        test_transform = label_encoder.inverse_transform(test_labels)
        logger.info("Label encoder compatibility test passed")
    except Exception as compat_error:
        logger.warning(f"Label encoder compatibility issue: {compat_error}")
        logger.info("You may need to retrain and save the label encoder with current sklearn version")
        
except Exception as e:
    logger.error(f"Error loading model or label encoder: {e}")
    model = None
    label_encoder = None

# Define emotions that indicate depression
depression_emotions = [
    'OAF_Sad', 'OAF_Fear', 'OAF_angry', 'OAF_disgust',
    'YAF_angry', 'YAF_disgust', 'YAF_fear', 'YAF_sad'
]

# Route to serve the frontend page
@app.route('/')
def index():
    return render_template("index.html")  # Loads index.html from frontend folder

# Function to extract MFCC features from audio
def extract_audio_features(file_path, sample_rate=22050):  # Changed to 22050 which is more common
    try:
        logger.info(f"Extracting features from: {file_path}")
        
        # Load audio file
        audio_data, sr = librosa.load(file_path, sr=sample_rate)
        logger.info(f"Audio loaded: duration={len(audio_data)/sr:.2f}s, sample_rate={sr}")
        
        # Check if audio is too short
        if len(audio_data) < sr * 0.5:  # Less than 0.5 seconds
            logger.warning("Audio file is too short")
            return None
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs = mfccs.T  # Transpose to get (time_steps, features)
        
        logger.info(f"MFCC features shape: {mfccs.shape}")
        return mfccs
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None

# Function to predict emotion from audio
def predict_emotion(file_path):
    try:
        # Check if model is loaded
        if model is None or label_encoder is None:
            logger.error("Model or label encoder not loaded")
            return None, None
            
        # Extract features
        features = extract_audio_features(file_path)
        if features is None:
            logger.error("Failed to extract features")
            return None, None

        logger.info(f"Original features shape: {features.shape}")
        
        # Get expected input shape from model
        expected_shape = model.input_shape
        logger.info(f"Model expects input shape: {expected_shape}")
        
        max_length = expected_shape[1]  # Time steps
        n_features = expected_shape[2]   # Number of features (should be 40 for MFCC)
        
        # Pad or truncate the time dimension
        if features.shape[0] < max_length:
            # Pad with zeros
            padding = ((0, max_length - features.shape[0]), (0, 0))
            features_padded = np.pad(features, padding, mode='constant', constant_values=0)
        else:
            # Truncate
            features_padded = features[:max_length, :]
            
        # Ensure feature dimension matches
        if features_padded.shape[1] != n_features:
            logger.error(f"Feature dimension mismatch: got {features_padded.shape[1]}, expected {n_features}")
            return None, None
            
        # Add batch dimension
        features_padded = np.expand_dims(features_padded, axis=0)
        logger.info(f"Final input shape: {features_padded.shape}")
        
        # Make prediction
        predictions = model.predict(features_padded, verbose=0)
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Prediction probabilities: {predictions[0]}")
        
        predicted_label = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)[0]
        
        logger.info(f"Predicted label index: {predicted_label[0]}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        # Convert to emotion name
        predicted_emotion = label_encoder.inverse_transform(predicted_label)[0]
        logger.info(f"Predicted emotion: {predicted_emotion}")
        
        # Determine depression status
        depression_status = "Depressed" if predicted_emotion in depression_emotions else "Not Depressed"
        logger.info(f"Depression status: {depression_status}")
        
        return predicted_emotion, depression_status, confidence
        
    except Exception as e:
        logger.error(f"Error in predict_emotion: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Route to handle audio upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    temp_path = None
    try:
        logger.info("Received prediction request")
        
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files['audio']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No file selected"}), 400
            
        logger.info(f"Processing file: {file.filename}")
        
        # Create temporary file with proper handling
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            # Write file data to temporary file
            with os.fdopen(temp_fd, 'wb') as temp_file:
                file.save(temp_file)
                
            # Check file size
            file_size = os.path.getsize(temp_path)
            logger.info(f"Temporary file size: {file_size} bytes")
            
            if file_size == 0:
                return jsonify({"error": "Empty audio file"}), 400
                
            # Make prediction
            result = predict_emotion(temp_path)
            
            if result[0] is None:
                return jsonify({"error": "Failed to process audio file"}), 500
            
            if len(result) == 3:
                predicted_emotion, depression_status, confidence = result
                response_data = {
                    "emotion": predicted_emotion, 
                    "status": depression_status,
                    "confidence": float(confidence)
                }
            else:
                predicted_emotion, depression_status = result
                response_data = {
                    "emotion": predicted_emotion, 
                    "status": depression_status
                }
                
            logger.info(f"Returning response: {response_data}")
            return jsonify(response_data)
            
        except Exception as inner_e:
            logger.error(f"Error processing audio: {inner_e}")
            return jsonify({"error": f"Failed to process audio: {str(inner_e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in predict route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
    finally:
        # Clean up temporary file safely
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Temporary file cleaned up successfully")
            except PermissionError:
                logger.warning(f"Could not delete temporary file: {temp_path}")
                # Try to delete later or ignore - not critical
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")

# Health check route
@app.route('/health')
def health_check():
    status = {
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "status": "healthy" if (model is not None and label_encoder is not None) else "unhealthy"
    }
    if model is not None:
        status["model_input_shape"] = model.input_shape
    if label_encoder is not None:
        status["available_emotions"] = label_encoder.classes_.tolist()
    return jsonify(status)

# Serve static files (CSS, JS, etc.)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)