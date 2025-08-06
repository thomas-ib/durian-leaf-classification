# 🍃 Durian Leaf Classifier API (Flask + ONNX)

This project is a lightweight, Dockerized Flask API that classifies durian tree leaves into one of several known varieties. It uses a fine-tuned ResNet18 model for image classification, converted to ONNX for fast CPU inference and reduced deployment size.

---

## 🧠 Model Details

### 📚 Architecture
- **Base Model**: `ResNet18` (pretrained on ImageNet via `torchvision.models`)
- **Modified Layer**:
  - Final `fc` layer changed from `nn.Linear(512, 1000)` → `nn.Linear(512, 4)`
- **Classes**:
  - `Bawor`
  - `DuriHitam`
  - `MusangKing`
  - `SuperTembaga`

### 🔧 Fine-Tuning Strategy
- **Backbone Layers**:
  - Layers `layer3`, `layer4`, and `fc` were **unfrozen**
  - Earlier layers were **frozen** to retain general visual features
- **Data Augmentation**:
  - `RandomResizedCrop`, `ColorJitter`, `RandomRotation`, `HorizontalFlip`
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Scheduler**: `ReduceLROnPlateau`
- **Early Stopping**: Monitored validation loss
- **Best Model Saved**: Based on lowest `val_loss`

### ✅ Model Performance

- **Dataset Size**: 1,530 images  
- **Train/Validation Split**: 80/20  
- **Validation Accuracy**: **~98%**  
- **Epochs Trained**: 6  
- **Peak Accuracy**: Achieved at epoch 5  
- The model shows strong generalization with minimal overfitting.

---

### 📂 Dataset

This project uses the **Durian Leaf** dataset provided on Kaggle:

> Dix2kurni. (2023). *Durian Leaf*. Kaggle. [https://www.kaggle.com/datasets/dix2kurni/durian-leaf](https://www.kaggle.com/datasets/dix2kurni/durian-leaf)

The dataset contains labeled images of durian leaf types, which were used for training and evaluation of the classification model.

---

## 🔁 Conversion to ONNX

The fine-tuned model was exported to ONNX using:

```python
torch.onnx.export(
    model,
    dummy_input,
    "durian_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
```

---

## 📦 Project Structure

```

.
├── durian\_classifier\_app.py     # Main Flask app (ONNX inference)
├── durian\_classifier.onnx       # Exported ONNX model
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Image build config
├── .dockerignore                # Files to exclude from Docker build
└── .github/workflows/deploy.yml# CI/CD pipeline to deploy to EC2

````

---

## 🚀 Running Locally (with Docker)

```bash
# Build the Docker image
docker build -t durian-classifier .

# Run the API on port 5000
docker run -p 5000:5000 durian-classifier
````

---

## 🔐 Token Authentication

The `/predict` endpoint is protected with a static token.

### Add a Header:

```http
Authorization: Bearer <your_api_token>
```

> Token is configured inside `durian_classifier_app.py`

---

## 🔍 API Usage

### `POST /predict`

**Headers**:

```
Authorization: Bearer <your_token>
```

**Body**: `form-data`

```
Key: image
Type: File
```

**Response**:

```json
{
  "prediction": "DuriHitam"
}
```

---

## ☁️ Deployment (CI/CD to AWS EC2)

* GitHub Actions pushes Docker images to Docker Hub
* SSHs into EC2 and runs the latest container
* Exposes API on port `80`

### `docker run` command on server:

```bash
docker run -d -p 80:5000 \
  --name durianclassifierapi \
  --cpus="0.5" \
  thomasibudiman/durian-classifier:vX
```

## 👨‍💻 Author

**Thomas Budiman**

* GitHub: [@thomas-ib](https://github.com/thomas-ib)
* Email: thomas.ibudiman@gmail.com

---

## 📝 License

MIT License – feel free to use, modify, and distribute.
