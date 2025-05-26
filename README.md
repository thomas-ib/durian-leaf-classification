# 🍃 Durian Leaf Classifier API

This project is a lightweight Flask API that classifies durian leaf images into one of several varieties using a fine-tuned `ResNet18` deep learning model. It has been optimized for deployment with Docker, secured with token-based access, and supports efficient inference using ONNX.

---

## 🧠 Model

- Backbone: `ResNet18`
- Trained with PyTorch and exported to ONNX
- Classifies images into:
  - `Bawor`
  - `DuriHitam`
  - `MusangKing`
  - `SuperTembaga`

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
