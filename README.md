# DogBreeds-Classifier

A machine learning web application that classifies dog breeds using a trained MobileNetV2 model with transfer learning.

## 🐕 Features

- **120 Dog Breeds Classification**: Trained on a comprehensive dataset of 120 different dog breeds
- **Web Interface**: Modern, responsive web application with drag-and-drop image upload
- **Real-time Predictions**: Instant breed predictions with confidence scores
- **Top 5 Predictions**: Shows the top 5 most likely breeds for each image
- **MobileNetV2 Architecture**: Uses transfer learning with MobileNetV2 for efficient and accurate predictions

## 🏗️ Architecture

- **Backend**: Flask web application
- **Frontend**: HTML5, CSS3, JavaScript with modern UI design
- **Model**: MobileNetV2 with transfer learning
- **Preprocessing**: Custom preprocessing pipeline matching training data
- **Deployment**: Ready for local deployment or cloud hosting

## 📁 Project Structure

```
DogBreeds-Classifier/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── templates/
│   └── index.html                  # Web interface
├── saved_models/                   # Saved model files (not in git)
│   ├── dog_breed_classifier_mobilenetv2.keras
│   ├── dog_breed_classifier_mobilenetv2.h5
│   └── dog_breed_classifier_mobilenetv2_class_names.json
├── Images/                         # Training dataset (not in git)
└── transfer_learning_mobilenet (1) (1).ipynb  # Training notebook
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.15+
- Flask
- PIL (Pillow)
- NumPy

### Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Aymanehajli/DogBreeds-Classifier.git
   cd DogBreeds-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model:**
   - The model files are not included in the repository due to size
   - You need to train the model using the notebook or download the pre-trained model
   - Place the model files in the `saved_models/` directory

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser and go to:**
   ```
   http://localhost:5003
   ```

## 🎯 Usage

1. **Upload an image**: Click the upload area or drag and drop a dog image
2. **Get predictions**: Click "Predict Breed" to get instant results
3. **View results**: See the predicted breed, confidence score, and top 5 predictions

## 📊 Model Information

- **Architecture**: MobileNetV2 with transfer learning
- **Input size**: 224x224 pixels
- **Classes**: 120 dog breeds
- **Training**: Transfer learning with fine-tuning
- **Accuracy**: ~83% on validation set

## 🔧 Training

To train your own model:

1. **Prepare dataset**: Organize images in `Images/` directory with breed subdirectories
2. **Run notebook**: Execute `transfer_learning_mobilenet (1) (1).ipynb`
3. **Save model**: The notebook will save the model in `saved_models/` directory

## 🛠️ Development

### Key Functions

- `load_and_preprocess_image()`: Loads and preprocesses images for prediction
- `predict_dog_breed_from_bytes()`: Predicts dog breed from image bytes
- `test_single_image()`: Tests model on a single image (from notebook)

### API Endpoints

- `GET /` - Main page with upload interface
- `POST /predict` - Upload image and get prediction

### Example API Response

```json
{
  "success": true,
  "predicted_breed": "golden_retriever",
  "confidence": 0.95,
  "top_5_predictions": [
    ["golden_retriever", 0.95],
    ["labrador_retriever", 0.03],
    ["flat_coated_retriever", 0.01],
    ["chesapeake_bay_retriever", 0.005],
    ["curly_coated_retriever", 0.005]
  ],
  "image_base64": "data:image/jpeg;base64,..."
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Make sure model files are in `saved_models/` directory
2. **Dependencies**: Install all requirements with `pip install -r requirements.txt`
3. **Port issues**: Change port in `app.py` if 5003 is already in use
4. **Image format**: Supports JPG, PNG, GIF (max 10MB)

### Debug Mode

Run with debug mode for detailed error messages:
```bash
python app.py
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 👨‍💻 Author

**Aymane Hajli**
- GitHub: [@Aymanehajli](https://github.com/Aymanehajli)

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- MobileNetV2 architecture for efficient transfer learning
- Flask for the web framework
- All contributors and testers 