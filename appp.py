from flask import Flask, request, jsonify, render_template
from scripts.utils import load_images_and_labels
from scripts.feature_extraction import extract_combined_features, extract_lbp_features
from scripts.model_training import train_knn_model
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset and train the model
print("Loading dataset and training model...")
dataset_folder = 'batik_dataset'
X, y = load_images_and_labels(dataset_folder)

# Extract combined features (LBP + custom features) for training
combined_features = [extract_combined_features(img) for img in X]
lbp_features = [extract_lbp_features(img) for img in X]
X_features = np.hstack([lbp_features, combined_features])

# Log total features during training
print(f"Total features during training: {X_features.shape[1]}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Train the KNN model
accuracy, knn_model = train_knn_model(X_scaled, y, metric='cosine', n_neighbors=5)
print(f"Model trained with accuracy: {accuracy:.2f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image and preprocess it
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        img = cv2.resize(img, (64, 64))  # Resize to match training size

        # Extract features for prediction (must match training process)
        lbp_features = extract_lbp_features(img)
        custom_features = extract_combined_features(img)
        features = np.hstack([lbp_features, custom_features]).reshape(1, -1)

        # Log the extracted feature size for debugging
        print(f"Extracted features during prediction: {features.shape[1]}")

        # Ensure feature dimensions match training
        if features.shape[1] != X_features.shape[1]:
            return jsonify({"error": f"Feature dimension mismatch! Expected {X_features.shape[1]}, got {features.shape[1]}"}), 500

        # Scale the features and predict
        features_scaled = scaler.transform(features)
        prediction = knn_model.predict(features_scaled)
        return jsonify({"predicted_motif": prediction[0]})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
