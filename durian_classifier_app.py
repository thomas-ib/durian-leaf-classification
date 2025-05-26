import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Bawor', 'DuriHitam', 'MusangKing', 'SuperTembaga']

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_resnet18_durian_leaf.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return 'Durian Classifier API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Received request")

    if 'image' not in request.files:
        print("🚫 No image key in request.files")
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        print("📁 Image file received:", file.filename)

        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # This might fail
        print("🖼️ Image opened successfully")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]
            print("✅ Prediction:", label)

        return jsonify({'prediction': label})
    
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)