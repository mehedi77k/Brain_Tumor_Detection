import os
import json
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from flask import Flask, render_template, request, jsonify
from PIL import Image
import matplotlib.cm as cm


# =====================================================
# Flask App Config
# =====================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_vgg16_best_fixed.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")

IMG_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


# =====================================================
# Load Model and Class Names
# =====================================================

print("Loading model...")

model = keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    idx_to_class = {int(v): k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    idx_to_class = {i: name for i, name in enumerate(class_names)}

print("Class names:", class_names)


# =====================================================
# Tumor Information
# =====================================================

TUMOR_INFO = {
    "glioma": {
        "display_name": "Glioma",
        "grade": "Grade IV - Malignant",
        "description": "Glioma is a type of brain tumor that starts in glial cells. Some gliomas can be aggressive and require urgent medical evaluation.",
        "recommendation": "Consult a neurologist or radiologist for detailed clinical confirmation."
    },
    "meningioma": {
        "display_name": "Meningioma",
        "grade": "Grade I-II - Usually Benign",
        "description": "Meningioma usually grows from the membranes surrounding the brain and spinal cord. Many cases are slow-growing but still require medical review.",
        "recommendation": "A doctor should review the MRI and decide whether monitoring or treatment is required."
    },
    "notumor": {
        "display_name": "No Tumor",
        "grade": "No visible tumor pattern",
        "description": "The model did not detect a tumor-like pattern from the uploaded MRI image.",
        "recommendation": "If symptoms exist, consult a doctor. AI output should not replace professional diagnosis."
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "grade": "Usually Non-cancerous",
        "description": "Pituitary tumors occur in the pituitary gland region. They may affect hormone regulation and vision depending on size and location.",
        "recommendation": "Consult an endocrinologist or neurologist for proper evaluation."
    }
}


# =====================================================
# Helper Functions
# =====================================================

def allowed_file(filename):
    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def preprocess_uploaded_image(file_storage):
    """
    Important:
    This preprocessing must match your training notebook.
    Your model was trained with VGG16 preprocess_input.
    So do not use image / 255.0 here.
    """

    image = Image.open(file_storage).convert("RGB")
    image_resized = image.resize(IMG_SIZE)

    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return image, image_resized, img_array


def decode_prediction(prediction):
    prediction = prediction[0]

    if prediction.shape[0] != len(class_names):
        raise ValueError(
            f"Model output size {prediction.shape[0]} does not match class count {len(class_names)}"
        )

    # If model output is not already softmax probability, apply softmax
    if (
        np.max(prediction) > 1
        or np.min(prediction) < 0
        or not np.isclose(np.sum(prediction), 1.0, atol=0.1)
    ):
        probabilities = tf.nn.softmax(prediction).numpy()
    else:
        probabilities = prediction

    predicted_index = int(np.argmax(probabilities))
    predicted_class = idx_to_class[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_dict = {
        idx_to_class[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(probabilities))
    }

    return predicted_class, confidence, probability_dict


def get_last_conv_layer_from_model():
    """
    Tries to find the last convolutional layer.
    Works for most VGG16 transfer learning models.
    """

    # Case 1: VGG16 is used as nested model inside Sequential
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            conv_layers = [
                sub_layer for sub_layer in layer.layers
                if isinstance(sub_layer, keras.layers.Conv2D)
            ]
            if conv_layers:
                return layer, conv_layers[-1]

    # Case 2: Conv layers are directly inside main model
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Conv2D)
    ]

    if conv_layers:
        return None, conv_layers[-1]

    return None, None


def generate_gradcam(img_array, predicted_index, original_resized_image):
    """
    Generates Grad-CAM heatmap overlay.
    If Grad-CAM fails for any reason, returns None.
    Prediction will still work.
    """

    try:
        nested_base_model, last_conv_layer = get_last_conv_layer_from_model()

        if last_conv_layer is None:
            print("No convolutional layer found for Grad-CAM.")
            return None

        # =====================================================
        # Case 1: Model has nested VGG16 base model
        # Example:
        # Sequential([
        #   VGG16(...),
        #   GlobalAveragePooling2D(),
        #   Dense(...)
        # ])
        # =====================================================

        if nested_base_model is not None:
            last_conv_model = keras.Model(
                nested_base_model.input,
                [last_conv_layer.output, nested_base_model.output]
            )

            classifier_input = keras.Input(shape=nested_base_model.output.shape[1:])
            x = classifier_input

            start_index = model.layers.index(nested_base_model) + 1

            for layer in model.layers[start_index:]:
                x = layer(x)

            classifier_model = keras.Model(classifier_input, x)

            with tf.GradientTape() as tape:
                conv_outputs, base_output = last_conv_model(img_array)
                tape.watch(conv_outputs)
                predictions = classifier_model(base_output)
                class_channel = predictions[:, predicted_index]

            grads = tape.gradient(class_channel, conv_outputs)

        # =====================================================
        # Case 2: Conv layers are directly inside main model
        # =====================================================

        else:
            grad_model = keras.Model(
                model.inputs,
                [last_conv_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_channel = predictions[:, predicted_index]

            grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            print("Gradients are None. Grad-CAM could not be generated.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)

        if max_value == 0:
            return None

        heatmap = heatmap / max_value
        heatmap = heatmap.numpy()

        # Resize heatmap to image size
        heatmap = np.uint8(255 * heatmap)
        heatmap_image = Image.fromarray(heatmap).resize(IMG_SIZE)

        heatmap_array = np.array(heatmap_image)

        # Apply colormap
        colormap = cm.get_cmap("jet")
        colored_heatmap = colormap(heatmap_array)
        colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)

        heatmap_pil = Image.fromarray(colored_heatmap)

        original_rgb = original_resized_image.convert("RGB")

        overlay = Image.blend(original_rgb, heatmap_pil, alpha=0.4)

        return pil_to_base64(overlay)

    except Exception as e:
        print("Grad-CAM error:", str(e))
        return None


def get_confidence_status(confidence_percent):
    if confidence_percent >= 85:
        return {
            "level": "High Confidence",
            "message": "The model is highly confident, but doctor review is still required."
        }
    elif confidence_percent >= 60:
        return {
            "level": "Moderate Confidence",
            "message": "The result should be reviewed carefully by a medical professional."
        }
    else:
        return {
            "level": "Low Confidence",
            "message": "The model is uncertain. Manual doctor review is strongly recommended."
        }


# =====================================================
# Routes
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "classes": class_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No selected file."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload JPG, PNG, JPEG, BMP, or WEBP image."
            }), 400

        original_image, resized_image, img_array = preprocess_uploaded_image(file)

        prediction = model.predict(img_array, verbose=0)

        predicted_class, confidence, probability_dict = decode_prediction(prediction)

        predicted_index = class_names.index(predicted_class)

        confidence_percent = round(confidence * 100, 2)

        tumor_info = TUMOR_INFO.get(predicted_class, {})

        gradcam_base64 = generate_gradcam(
            img_array=img_array,
            predicted_index=predicted_index,
            original_resized_image=resized_image
        )

        confidence_status = get_confidence_status(confidence_percent)

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "display_name": tumor_info.get("display_name", predicted_class.title()),
            "confidence": confidence_percent,
            "probabilities": probability_dict,
            "grade": tumor_info.get("grade", "N/A"),
            "description": tumor_info.get("description", ""),
            "recommendation": tumor_info.get("recommendation", ""),
            "confidence_status": confidence_status,
            "gradcam": gradcam_base64
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =====================================================
# Run App
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)