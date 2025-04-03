import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Get the absolute path to the directory the script is running in
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')

# Create templates directory if it doesn't exist
os.makedirs(template_dir, exist_ok=True)

# Initialize Flask with explicit template directory
app = Flask(__name__, template_folder=template_dir)

# Define the upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {base_dir}")
print(f"Template directory: {template_dir}")
print(f"Template directory exists: {os.path.exists(template_dir)}")
if os.path.exists(template_dir):
    print(f"Contents of templates folder: {os.listdir(template_dir)}")


# Function to load models safely
def load_model_safe(model_path, model_type="joblib"):
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found. Skipping...")
        return None
    print(f"Loading model: {model_path}")
    return joblib.load(model_path) if model_type == "joblib" else load_model(model_path)


# Load models
mp_neuron_model = load_model_safe("mp_neuron_model.pkl", "joblib")
cnn_model = load_model_safe("my_cnn_model.h5", "keras")
feature_extractor_model = load_model_safe("feature_extractor.h5", "keras")
scaler = load_model_safe("scaler.pkl", "joblib")
pca = load_model_safe("pca.pkl", "joblib")

# Load class labels
try:
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
except FileNotFoundError:
    print("Warning: Class labels file 'class_labels.json' not found. Skipping...")
    class_labels = {}

# Create results.html file if it doesn't exist
results_template_path = os.path.join(template_dir, 'results.html')
if not os.path.exists(results_template_path):
    print(f"Creating results.html template at {results_template_path}")
    with open(results_template_path, 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disease Analysis Results</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4CAF50;
            --secondary-color: #2E7D32;
            --accent-color: #8BC34A;
            --dark-bg: rgba(0, 0, 0, 0.7);
            --light-text: #fff;
            --warning-color: #FFC107;
            --danger-color: #F44336;
            --success-color: #4CAF50;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-image: url('{{ url_for("static", filename="images/Background.jpg") }}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            padding: 40px 20px;
            position: relative;
        }

        body::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: -1;
        }

        .container {
            width: 90%;
            max-width: 1000px;
            margin: 0 auto;
            color: var(--light-text);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            color: var(--accent-color);
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        .header p {
            font-size: 1.2rem;
            color: #ddd;
        }

        .result-container {
            display: flex;
            flex-direction: column;
            background-color: var(--dark-bg);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.5);
            margin-bottom: 30px;
        }

        .result-header {
            background-color: rgba(0, 0, 0, 0.4);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .result-header h2 {
            font-size: 1.8rem;
            color: var(--accent-color);
        }

        .result-content {
            display: flex;
            flex-direction: column;
            padding: 0;
        }

        @media (min-width: 768px) {
            .result-content {
                flex-direction: row;
            }
        }

        .image-section {
            flex: 1;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: rgba(0, 0, 0, 0.2);
        }

        .leaf-image {
            max-width: 100%;
            height: auto;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            border: 3px solid var(--accent-color);
        }

        .details-section {
            flex: 1.2;
            padding: 30px;
        }

        .diagnosis-box {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .diagnosis-title {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }

        .diagnosis-title i {
            font-size: 24px;
            margin-right: 10px;
            color: var(--accent-color);
        }

        .diagnosis-title h3 {
            font-size: 1.4rem;
        }

        .diagnosis-item {
            margin-bottom: 15px;
        }

        .diagnosis-item h4 {
            font-size: 1.1rem;
            margin-bottom: 5px;
            color: #ddd;
        }

        .diagnosis-value {
            font-size: 1.3rem;
            font-weight: bold;
            color: var(--accent-color);
        }

        .confidence-meter {
            height: 10px;
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-level {
            height: 100%;
            background-color: var(--accent-color);
            border-radius: 5px;
        }

        .treatment-box {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
        }

        .treatment-content {
            line-height: 1.6;
        }

        .actions {
            display: flex;
            justify-content: center;
            margin-top: 30px;
            gap: 20px;
        }

        .action-btn {
            padding: 12px 25px;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            text-decoration: none;
            text-align: center;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .action-btn i {
            margin-right: 8px;
        }

        .primary-btn {
            background-color: var(--primary-color);
            color: white;
        }

        .primary-btn:hover {
            background-color: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .secondary-btn {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
        }

        .secondary-btn:hover {
            background-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            color: #aaa;
            font-size: 0.9rem;
        }

        /* Disease specific colors */
        .healthy {
            --accent-color: #4CAF50;
        }

        .mild {
            --accent-color: #FFC107;
        }

        .severe {
            --accent-color: #F44336;
        }

        /* Severity indicator */
        .severity-indicator {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }

        .severity-dot {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 5px;
        }

        .severity-dot.healthy {
            background-color: var(--success-color);
        }

        .severity-dot.mild {
            background-color: var(--warning-color);
        }

        .severity-dot.severe {
            background-color: var(--danger-color);
        }

        .severity-text {
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Disease Analysis Results</h1>
            <p>Detailed analysis of your pumpkin leaf image</p>
        </div>

        <div class="result-container">
            <div class="result-header">
                <h2>Leaf Analysis Complete</h2>
            </div>

            <div class="result-content">
                <div class="image-section">
                    <img src="{{ image_url }}" alt="Analyzed Leaf" class="leaf-image">
                </div>

                <div class="details-section">
                    <div class="diagnosis-box">
                        <div class="diagnosis-title">
                            <i class="fas fa-search-plus"></i>
                            <h3>Diagnosis</h3>
                        </div>

                        <div class="diagnosis-item">
                            <h4>Identified Condition:</h4>
                            <div class="diagnosis-value">{{ prediction }}</div>

                            {% if "Healthy" in prediction %}
                            <div class="severity-indicator">
                                <div class="severity-dot healthy"></div>
                                <div class="severity-text">Healthy plant</div>
                            </div>
                            {% elif "Powdery_Mildew" in prediction or "Mosaic_Disease" in prediction %}
                            <div class="severity-indicator">
                                <div class="severity-dot mild"></div>
                                <div class="severity-text">Moderate concern</div>
                            </div>
                            {% else %}
                            <div class="severity-indicator">
                                <div class="severity-dot severe"></div>
                                <div class="severity-text">Requires immediate attention</div>
                            </div>
                            {% endif %}
                        </div>

                        <div class="diagnosis-item">
                            <h4>Confidence Level:</h4>
                            <div class="diagnosis-value">{{ confidence }}</div>
                            <div class="confidence-meter">
                                <div class="confidence-level" style="width: {{ confidence }}"></div>
                            </div>
                        </div>
                    </div>

                    <div class="treatment-box">
                        <div class="diagnosis-title">
                            <i class="fas fa-medkit"></i>
                            <h3>Recommended Treatment</h3>
                        </div>

                        <div class="treatment-content">
                            {{ treatment }}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="actions">
            <a href="/upload" class="action-btn secondary-btn">
                <i class="fas fa-upload"></i> Analyze Another Leaf
            </a>
            <a href="/" class="action-btn primary-btn">
                <i class="fas fa-home"></i> Back to Home
            </a>
        </div>

        <div class="footer">
            <p>© 2025 Pumpkin Leaf Disease Detector | AI-Powered Plant Health Monitoring</p>
        </div>
    </div>
</body>
</html>''')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        # Save the file
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)

        # Process the image and get prediction
        try:
            predicted_label, confidence, treatment = process_image(filename)

            # Create relative path for image URL
            image_url = os.path.join("uploads", os.path.basename(filename))

            # Redirect to results page with query parameters
            return redirect(url_for('results',
                                    image=image_url,
                                    prediction=predicted_label,
                                    confidence=confidence,
                                    treatment=treatment))

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return render_template("upload.html", error=f"Error processing image: {str(e)}")

    return render_template("upload.html")


@app.route("/results")
def results():
    # Get parameters from the URL
    image = request.args.get('image', '')
    prediction = request.args.get('prediction', '')
    confidence = request.args.get('confidence', '')
    treatment = request.args.get('treatment', '')

    # Create the full image path for the template
    image_path = url_for('static', filename=image)

    # Debug: print template information
    print(f"Rendering results.html with: {image_path}, {prediction}, {confidence}")
    print(f"Looking for template at: {os.path.join(template_dir, 'results.html')}")
    print(f"Template exists: {os.path.exists(os.path.join(template_dir, 'results.html'))}")

    try:
        return render_template("results.html",
                               image_url=image_path,
                               prediction=prediction,
                               confidence=confidence,
                               treatment=treatment)
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        # Fallback response if template not found
        return f"""
        <html>
        <head><title>Results</title></head>
        <body>
            <h1>Results (Fallback Page)</h1>
            <p>The template file could not be found. Please check your project structure.</p>
            <p>Image: {image_path}</p>
            <p>Prediction: {prediction}</p>
            <p>Confidence: {confidence}</p>
            <p>Treatment: {treatment}</p>
            <p>Error: {str(e)}</p>
            <a href="/upload">Try again</a>
        </body>
        </html>
        """


def process_image(filepath):
    # Load and preprocess image
    image = load_img(filepath, target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Extract CNN Features
    if not feature_extractor_model:
        raise Exception("Feature extractor model not available!")

    features = feature_extractor_model.predict(image_array)
    features = features.reshape(1, -1)

    # Apply Standardization & PCA
    if not (scaler and pca):
        raise Exception("Scaler or PCA model not available!")

    features = scaler.transform(features)
    features = pca.transform(features)

    # Make Prediction
    if not mp_neuron_model:
        raise Exception("MP Neuron model not available!")

    prediction_raw = mp_neuron_model.predict(features)
    prediction_index = int(np.round(prediction_raw[0]))

    # Get prediction label
    predicted_label = class_labels.get(str(prediction_index), "Unknown Class")
    if predicted_label == "Unknown Class":
        raise Exception(f"Invalid prediction index: {prediction_index}")

    # Get treatment and confidence
    treatment = get_treatment(predicted_label)
    confidence = "95%"  # Placeholder for real confidence

    return predicted_label, confidence, treatment


# Get treatment suggestions
def get_treatment(disease):
    treatments = {
        "Powdery_Mildew": "Apply sulfur-based fungicide or potassium bicarbonate solution (1 tablespoon per gallon of water). Improve air circulation by pruning excess foliage and avoid overhead watering to reduce humidity.",
        "Mosaic_Disease": "Remove and destroy infected plants to prevent the virus from spreading. Control aphids and other insect vectors using insecticidal soap or neem oil, as they transmit the virus. Plant virus-resistant pumpkin varieties and practice crop rotation to reduce the risk of infection in future crops.",
        "Downy_Mildew": "Apply fungicides such as copper-based sprays or chlorothalonil early to prevent the disease from spreading. Improve air circulation by spacing plants properly, remove weeds, and avoid overhead watering to reduce moisture buildup which favors fungal growth.",
        "Bacterial_Spot": "Use copper fungicides as a preventive measure. Apply copper fungicide spray before or after rain (not during). If signs appear, spray for 7-10 days, then continue weekly after transplanting. Maintenance treatments every 10 days in dry weather and 5-7 days in rainy weather.",
        "Healthy": "No treatment needed. Maintain proper watering and sunlight exposure. Ensure adequate spacing between plants for good air circulation and conduct periodic soil health checks."
    }

    # Handle exact match case
    if disease in treatments:
        return treatments[disease]

    # Handle case variations (e.g., spaces instead of underscores)
    disease_normalized = disease.replace(" ", "_")
    if disease_normalized in treatments:
        return treatments[disease_normalized]

    # Handle partial matches
    for key in treatments.keys():
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return treatments[key]

    return "No treatment recommendation available for this condition."


if __name__ == '__main__':
    app.run(debug=True)