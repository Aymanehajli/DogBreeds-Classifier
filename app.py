from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import base64

app = Flask(__name__)

# Load the trained model and class names
def load_model():
    """Load the trained model and class names"""
    try:
        # Load class names
        with open("saved_models/dog_breed_classifier_mobilenetv2_class_names.json", 'r') as f:
            class_names = json.load(f)
        
        # Load model 
        keras_model_path = "saved_models/dog_breed_classifier_mobilenetv2.keras"
        h5_model_path = "saved_models/dog_breed_classifier_mobilenetv2.h5"
        
        if os.path.exists(keras_model_path):
            model = tf.keras.models.load_model(keras_model_path)
            print(f"✅ Model loaded from: {keras_model_path}")
        elif os.path.exists(h5_model_path):
            model = tf.keras.models.load_model(h5_model_path)
            print(f"✅ Model loaded from: {h5_model_path}")
        else:
            raise FileNotFoundError("No saved model found!")
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Load model globally
model, class_names = load_model()

def load_and_preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction
    (Match the training data: pixel values in 0-255 float32, no normalization)
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to float32 array WITHOUT dividing by 255
    img_array = np.array(img).astype('float32')
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_dog_breed_from_bytes(image_bytes):
   
    if model is None:
        print("Model not loaded!")
        return None, None, None
    
    try:
        # Load and preprocess image using the notebook function
        img_array = load_and_preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_breed = class_names[predicted_class]
        
        # Get top 5 predictions 
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_breeds = [class_names[i] for i in top_5_indices]
        top_5_confidences = np.array([predictions[0][i] for i in top_5_indices])
        top_5_confidences = top_5_confidences / np.sum(top_5_confidences) 
        
        # Convert numpy types to Python native types for JSON serialization
        confidence = float(confidence)
        top_5_predictions = [(breed, float(conf)) for breed, conf in zip(top_5_breeds, top_5_confidences)]
        
        print(f"Prediction successful: {predicted_breed} ({confidence:.2%})")
        return predicted_breed, confidence, top_5_predictions
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Make prediction using the notebook function
        predicted_breed, confidence, top_5_predictions = predict_dog_breed_from_bytes(image_bytes)
        
        if predicted_breed is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'predicted_breed': predicted_breed,
            'confidence': confidence,  
            'top_5_predictions': top_5_predictions,  
            'image_base64': image_base64
        })
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003) 