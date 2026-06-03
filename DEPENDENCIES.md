# Project Dependencies

This document describes the runtime and tooling dependencies used across the project.

## Runtime (Python)

These packages are required to run the Flask web app and inference pipeline. They are listed in requirements.txt.

- Flask: Web server, routing, and request handling for the UI and API endpoints.
- TensorFlow: Model loading and inference for the VGG16-based classifier.
- NumPy: Array operations used in preprocessing and postprocessing.
- Pillow: Image loading and resizing for MRI inputs.
- Matplotlib: Grad-CAM rendering and heatmap overlay generation.
- h5py: HDF5 support for loading .h5 model files.
- gunicorn: Production-ready WSGI server (used for deployment, not required for local dev).

## Frontend Libraries

- Chart.js: Client-side probability chart visualization.

## System Requirements

- Python 3.9+ recommended.
- A TensorFlow-compatible runtime (CPU or GPU). GPU is optional.

## Notes

- The app expects the trained model file at model/brain_tumor_vgg16_best_fixed.h5.
- Install all Python dependencies via:

```bash
pip install -r requirements.txt
```
