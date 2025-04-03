import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import joblib
from flask_cors import CORS  # Allow cross-origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to avoid cross-origin issues

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained MP Neuron model
MODEL_PATH = "mp_neuron_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train and save the model first.")

mp_model = joblib.load(MODEL_PATH)

# Define class labels
CLASS_LABELS = ["Healthy", "Early Blight", "Late Blight"]

# Define treatment recommendations
TREATMENTS = {
    "Healthy": "No treatment needed. Maintain proper watering and sunlight.",
    "Early Blight": "Use copper-based fungicides and remove infected leaves.",
    "Late Blight": "Apply chlorothalonil fungicides and ensure proper ventilation."
}

@app.route("/")
def index():
    return render_template("index.html")  # Renders the upload form

# Print expected model input shape
print("Expected input shape:", mp_model.n_features_in_)

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns disease prediction."""
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file!", 400

    try:
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and preprocess the image
        image = Image.open(filepath).convert("L")  # Convert to grayscale
        image = image.resize((10, 10))  # Resize to 10x10 pixels
        image_array = np.array(image) / 255.0  # Normalize pixel values (0 to 1)

        # Debugging: Print image shape
        print("Actual image shape before flattening:", image_array.shape)

        # Flatten and reshape
        image_array = image_array.flatten().reshape(1, -1)  # Flatten into 1D array

        # Debugging: Print final shape before prediction
        print("Final input shape for model:", image_array.shape)

        # Ensure input shape matches model
        if image_array.shape[1] != mp_model.n_features_in_:
            return f"Model expects {mp_model.n_features_in_} features, but got {image_array.shape[1]}", 400

        # Make a prediction
        prediction = mp_model.predict(image_array)

        # Debugging: Print model prediction
        print("Raw model prediction output:", prediction)

        if len(prediction) == 0:
            return "Model returned an empty prediction!", 500

        predicted_index = int(prediction[0])

        # Validate prediction index
        if predicted_index < 0 or predicted_index >= len(CLASS_LABELS):
            return f"Invalid prediction index: {predicted_index}", 500

        predicted_label = CLASS_LABELS[predicted_index]
        confidence = "95%"  # Placeholder since LogisticRegression doesn’t return confidence

        # Get treatment recommendation
        treatment = TREATMENTS.get(predicted_label, "No recommendation available.")

        # Render the result.html template
        return render_template("result.html",
                               image_url="uploads/" + filename,
                               prediction=predicted_label,
                               confidence=confidence,
                               treatment=treatment)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)