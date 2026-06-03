# NeuraScan - Brain Tumor Detection

NeuraScan is a Flask web app that classifies brain MRI images into four categories (glioma, meningioma, pituitary, or no tumor). It also generates Grad-CAM explanations and a probability chart for transparent, educational demonstrations.

## Features

- Drag-and-drop MRI upload with preview
- VGG16-based classifier with class probabilities
- Grad-CAM heatmap overlay for explainability
- Medical info panel with grade, description, and recommendation
- Lightweight single-page UI

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
- model/ - Trained model, class index mapping, notebooks

## Requirements

- Python 3.9+ recommended
- CPU or GPU TensorFlow runtime supported by your environment

## Setup

1) Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

## API

### GET /health

Returns server status, model readiness, and class names.

### POST /predict

Form field:

- image (file upload)

Response fields:

- predicted class
- confidence
- class probabilities
- Grad-CAM image (base64)
- medical info fields

Example:

```bash
curl -X POST -F image=@path/to/scan.jpg http://127.0.0.1:5000/predict
```

## Model Notes

- Model file: model/brain_tumor_vgg16_best_fixed.h5
- Input size: 224 x 224
- Preprocessing: VGG16 preprocess_input (do not normalize with /255.0)

## Troubleshooting

- If the model fails to load, confirm the file exists and matches the expected name.
- If predictions look incorrect, verify preprocessing matches VGG16 requirements.
- For slow inference, ensure you are not running other heavy processes or switch to a GPU runtime.

## Disclaimer

This project is for educational and research demonstration only. It is not a substitute for professional medical diagnosis. Always consult qualified medical experts for clinical decisions.
