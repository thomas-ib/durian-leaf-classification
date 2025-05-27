import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import onnxruntime as ort
import numpy as np
from flask_cors import CORS

API_TOKEN = "TEST_T0KEN_101000"

app = Flask(__name__)

CORS(app,
     origins=["http://localhost:5173"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"])


class_names = ['Bawor', 'DuriHitam', 'MusangKing', 'SuperTembaga']

# Load ONNX model
session = ort.InferenceSession("durian_classifier.onnx", providers=["CPUExecutionProvider"])

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check for Authorization header
    token = request.headers.get('Authorization')
    if token != f"Bearer {API_TOKEN}":
        return jsonify({'error': 'Unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # This might fail
        input_tensor = transform(image).unsqueeze(0).numpy()

        outputs = session.run(None, {"input": input_tensor})
        prediction = class_names[np.argmax(outputs[0])]

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)