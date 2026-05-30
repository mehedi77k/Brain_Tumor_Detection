# NeuraScan - Brain Tumor Detection

NeuraScan is a Flask-based web application that classifies brain MRI images into four categories (glioma, meningioma, pituitary, or no tumor). It provides Grad-CAM explainability and a probability chart to support transparent, educational demonstrations.

## Highlights

- MRI image upload with preview
- VGG16-based classifier with class probabilities
- Grad-CAM heatmap overlay for explainability
- Medical info panel with grade, description, and recommendation

## Tech Stack

- Python, Flask
- TensorFlow / Keras
- NumPy, Pillow, Matplotlib
- Chart.js for probability visualization

## Project Structure

- app.py - Flask server, model loading, prediction, and Grad-CAM
- templates/index.html - UI layout
- static/css/style.css - UI styling
- static/js/app.js - Frontend logic and chart rendering
- model/ - Trained model and class index mapping

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## API

### GET /health

Returns server status, model readiness, and class names.

### POST /predict

Form field:
- image (file upload)

Response:
- predicted class
- confidence
- class probabilities
- Grad-CAM image (base64)
- medical info fields

Example request:

```bash
curl -X POST -F image=@path/to/scan.jpg http://127.0.0.1:5000/predict
```

## Model Notes

- Model file: model/brain_tumor_vgg16_best_fixed.h5
- Input size: 224 x 224
- Preprocessing: VGG16 preprocess_input (do not normalize with /255.0).

## Disclaimer

This project is for educational and research demonstration only. It is not a substitute for professional medical diagnosis. Always consult qualified medical experts for clinical decisions.
