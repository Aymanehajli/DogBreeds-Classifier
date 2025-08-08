import tensorflow as tf
import numpy as np
import json
import os

def debug_model():
    
    
    try:
        # Load class names
        with open("saved_models/dog_breed_classifier_mobilenetv2_class_names.json", 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} class names")
        
        # Load model
        keras_model_path = "saved_models/dog_breed_classifier_mobilenetv2.keras"
        if os.path.exists(keras_model_path):
            model = tf.keras.models.load_model(keras_model_path)
            print(f"Model loaded from: {keras_model_path}")
        else:
            print("Model file not found!")
            return
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        # Check model layers
        print("\n🔍 Model Layers:")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} - {type(layer).__name__}")
        
        # Test with a simple image
        print("\n🧪 Testing with dummy image...")
        
        # Create a dummy image (224x224x3)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Preprocess
        dummy_image = tf.keras.applications.mobilenet_v2.preprocess_input(dummy_image)
        dummy_image = np.expand_dims(dummy_image, axis=0)
        
        print(f"Dummy image shape: {dummy_image.shape}")
        
        # Try prediction
        try:
            predictions = model(dummy_image, training=False)
            print(f"Prediction successful! Shape: {predictions.shape}")
            
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_breed = class_names[predicted_class]
            
            print(f"Predicted: {predicted_breed} ({confidence:.2%})")
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model() 