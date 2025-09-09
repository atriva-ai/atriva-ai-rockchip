
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

### **3️⃣ Verify NPU Hardware**
Before running the API, verify that your RKNPU hardware is working:
```sh
python3 test_rknpu.py
```

**Prerequisites:**
- Official OS image from http://www.orangepi.org was installed.
- RKNPU kernel driver must be loaded (built into kernel)
- `librknnrt.so` runtime library must be present
- RKNN toolkit2 must be installed (included in requirements.txt)

### **4️⃣ Run the API Locally**
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


## ✅ Verifying RKNPU Readiness on RK3588 (Orange Pi 5 / 5 Plus)

To confirm that the Rockchip NPU is enabled and ready for inference on Ubuntu images:

---

### 1. Check Kernel Driver
The RKNPU driver is usually **built into the kernel**, not a loadable module (so it will not appear in `lsmod`).

```bash
dmesg | grep -i rknpu
````

Expected output (example):

```
RKNPU fdab0000.npu: RKNPU: rknpu iommu is enabled, using iommu mode
[drm] Initialized rknpu 0.9.6 ...
```

Driver version:

```bash
sudo cat /sys/kernel/debug/rknpu/version
```

Expected:

```
RKNPU driver: v0.9.6
```

---

### 2. Verify Runtime Library

Ensure that `librknnrt.so` is installed:

```bash
ls -l /usr/lib/ | grep librknnrt
strings /usr/lib/librknnrt.so | grep "librknnrt version"
```

Expected:

```
librknnrt.so
librknnrt version: 2.3.2 (...)
```

---

### 3. Install Python API

The runtime toolkit is needed for Python testing:

```bash
pip install rknn-toolkit-lite2
```

---

### 4. Run Sanity Test

Download a sample model and test inference on the NPU:

```bash
wget https://github.com/airockchip/rknn-toolkit2/raw/master/examples/mobilenet_v1/mobilenet_v1.rknn
```

Create `test_rknpu.py`:

```python
from rknnlite.api import RKNNLite
import numpy as np

print("=== RKNN NPU Sanity Test ===")
rknn = RKNNLite()

# Load model
ret = rknn.load_rknn('mobilenet_v1.rknn')
if ret != 0:
    print("❌ Failed to load model")
    exit(1)

# Init runtime
ret = rknn.init_runtime()
if ret != 0:
    print("❌ Failed to init runtime")
    exit(1)

print("✅ NPU runtime initialized")

# Run inference
dummy = np.random.randint(0, 255, (1,224,224,3), dtype=np.uint8)
outputs = rknn.inference(inputs=[dummy])
print("✅ Inference ran successfully, output shapes:", [o.shape for o in outputs])

rknn.release()
```


### 5. **NPU Hardware Test**
Test RKNPU hardware functionality without using the API service:
```sh
python3 test_rknpu.py
```

This test will:
- ✅ Verify RKNN toolkit installation
- ✅ Check for required runtime libraries (`librknnrt.so`)
- ✅ Test RKNN API import and object creation
- ✅ Convert ONNX model to RKNN format (if needed)
- ✅ Initialize NPU runtime and verify hardware functionality
- ✅ Confirm NPU is ready for inference

**Expected Output:**
```
RKNPU Standalone Test
==================================================
=== RKNPU Hardware Test ===
✅ librknnrt.so found
✅ RKNN API imported successfully
✅ RKNN object created successfully
✅ Found ONNX model, converting to RKNN: ./models/yolov8n.onnx
✅ RKNN configured successfully
✅ ONNX model loaded successfully
✅ RKNN model built successfully
✅ NPU runtime initialized successfully
   🎉 RKNPU hardware is working!
✅ Created dummy input data
✅ NPU runtime is functional and ready for inference
✅ RKNN resources released

=== Model Conversion Test ===
✅ Found ONNX models: ['yolov8n.onnx']
✅ RKNN conversion API available

==================================================
TEST SUMMARY:
Hardware Test: ✅ PASS
Conversion Test: ✅ PASS

🎉 All tests passed! RKNPU is ready for use.
```


### 5. Notes

* `lsmod` will **not show `rknpu`** on RK3588 because the driver is compiled into the kernel (`CONFIG_ROCKCHIP_NPU=y`).
* Check `/sys/kernel/debug/rknpu/` for driver info and stats.
* Ensure the Ubuntu image you are using includes Rockchip patches for the NPU.

---

## **📜 License**
This project is licensed under the **MIT License**.

