
## **📝 README.md (Atriva AI API)**

```md
# Atriva AI API with RKNN for Rockchip 🚀

This is a FastAPI-based AI API that leverages **RKNN (Rockchip Neural Network)** for optimized deep learning inference on Rockchip hardware.  
It provides a RESTful interface for running AI models, such as object detection and image classification on RK3588 platform.

## **📂 Project Structure**
```plaintext
atriva-ai-rockchip/
│── app/
│   ├── routes.py         # API route definitions
│   ├── services.py       # AI model processing logic
│   ├── models.py         # Model management and RKNN conversion
│   ├── shared_data.py    # Shared data utilities
│── rknpu/                # RKNN runtime files
│   ├── librknnrt.so     # RKNN runtime library
│   ├── rknn_server      # RKNN server binary
│   ├── start_rknn.sh    # RKNN server startup script
│── models/               # Downloaded and converted RKNN models
│── main.py               # Entry point for FastAPI
│── requirements.txt      # Python dependencies
│── Dockerfile            # Docker configuration
│── .dockerignore         # Ignore unnecessary files in Docker builds
│── README.md             # Project documentation
│── .gitignore            # Ignore unnecessary files in Git
```

## **⚡ Features**
✅ FastAPI-based AI API  
✅ RKNN optimization for Rockchip RK3588  
✅ Automatic ONNX to RKNN conversion  
✅ Hardware acceleration with NPU  
✅ Dockerized for easy deployment  
✅ Object detection with model mapping  
✅ Direct model inference support  

## **🔧 Setup & Installation**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/atriva-ai/atriva-ai-rockchip.git
cd atriva-ai-rockchip
```

### **2️⃣ Create a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3️⃣ Run the API Locally**
```sh
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
Access the API documentation at:  
👉 **http://localhost:8001/docs**

## **🐳 Running with Docker**
### **1️⃣ Build the Docker Image**
```sh
docker build -t atriva-ai-rockchip .
```

### **2️⃣ Run the Container**
```sh
docker run -d -p 8001:8001 --name ai-rockchip-container atriva-ai-rockchip
```
Now, visit:  
👉 **http://localhost:8001/docs**

## **🛠 API Endpoints**

### **Object Detection**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/objects` | List available object types |
| `POST` | `/inference/detection` | Run object detection |

### **Direct Model Inference**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List available models |
| `POST` | `/inference/direct` | Run direct model inference |
| `POST` | `/model/load` | Load a specific model |

### **Camera Frame Processing**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/shared/cameras` | List available cameras |
| `GET` | `/shared/cameras/{id}/frames` | Get camera frame info |
| `POST` | `/shared/cameras/{id}/inference` | Run inference on camera frame |

## **🎯 Supported Models**

### **Object Detection Models**
- **YOLOv8n**: General object detection
- **YOLOv8n-pose**: Human pose estimation
- **YOLOv8n-obb**: Oriented bounding box detection
- **YOLOv11n**: Latest YOLO detection

### **Specialized Models**
- **RetinaFace-mobile-320**: Face detection
- **LPRNet**: License plate recognition
- **CLIP**: Image and text understanding
- **YAMNet**: Audio classification

## **🔧 Object Mapping**

The API provides user-friendly object names that map to specific models:

```python
# Object Detection (User-Friendly)
detections = run_object_detection(image, "human")      # Uses YOLOv8n
detections = run_object_detection(image, "face")       # Uses RetinaFace
detections = run_object_detection(image, "license-plate") # Uses LPRNet
detections = run_object_detection(image, "pose")       # Uses YOLOv8n-pose

# Direct Model Inference (Advanced)
output = run_inference(image, "yolov8n")              # Direct model access
output = run_inference(image, "LPRNet")               # Direct model access
```

## **🧪 Running Tests**
```sh
pytest tests/
```

## **📜 License**
This project is licensed under the **MIT License**.

