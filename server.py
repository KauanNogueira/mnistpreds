import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from model import LinearClassifierModel

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "models"
loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        print(f"Loading model '{model_name}' for the first time...")
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        INPUT_FEATURES = 28 * 28
        OUTPUT_FEATURES = 10
        
        model = LinearClassifierModel(input_feat=INPUT_FEATURES, output_feat=OUTPUT_FEATURES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        loaded_models[model_name] = model
        print(f"Model '{model_name}' loaded and cached.")
    
    return loaded_models[model_name]

@app.route("/models", methods=["GET"])
def list_models():
    try:
        files = os.listdir(MODELS_DIR)
        model_files = [f for f in files if f.endswith(".pth")]
        return jsonify(model_files)
    except FileNotFoundError:
        return jsonify({"error": f"Models directory not found at '{MODELS_DIR}'"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    image_data = data.get('image')
    model_name = data.get('model_name')

    if not image_data:
        return jsonify({"error": "Image data not found"}), 400
    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400

    try:
        model = get_model(model_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Error loading model: {e}"}), 500

    try:
        image_tensor = torch.tensor(image_data, dtype=torch.float32)
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.reshape(1, 1, 28, 28)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 400

    with torch.inference_mode():
        output_logits = model(image_tensor)
        
        # Apply Softmax to get probabilities
        probabilities = torch.softmax(output_logits, dim=1)
        
        # Get the top prediction and its probability
        pred_prob, pred_label_tensor = torch.max(probabilities, dim=1)
        
        pred_label = pred_label_tensor.item()

    # Return the final prediction and the full list of probabilities
    return jsonify({
        "prediction": pred_label,
        "probabilities": probabilities.tolist()[0]
    })

@app.route("/weights", methods=["GET"])
def get_weights():
    model_name = request.args.get('model_name')
    digit_str = request.args.get('digit')

    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400
    if not digit_str:
        return jsonify({"error": "Digit not specified"}), 400

    try:
        digit = int(digit_str)
        if not 0 <= digit <= 9:
            raise ValueError("Digit must be between 0 and 9")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        model = get_model(model_name)
        weights = model.linear_layer.weight[digit]
        return jsonify(weights.tolist())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500



if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        print(f"Error: The '{MODELS_DIR}' directory does not exist. Please create it and add model files.")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
