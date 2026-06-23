# DeepLabV3+ FastAPI Server Documentation

## Overview

This FastAPI server provides REST endpoints for farmland boundary segmentation using a DeepLabV3+ ResNet34 ONNX model. It accepts satellite tile images, performs semantic segmentation, extracts boundary contours, and returns GeoJSON polygons.

## Architecture

```
Satellite Tile (.jpg/.png)
         ↓
  [FastAPI Server]
         ↓
  ONNX Inference (768×768)
         ↓
  Mask → Contours → Polygons
         ↓
  GeoJSON Output
```

---

## Quick Start

### 1. Installation

```bash
cd "d:\farmland boundary\New folder\fastapi_server"
pip install -r requirements.txt
```

### 2. Start Server

```bash
python main.py
```

Server runs on: `http://localhost:8000`

### 3. Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### ✅ Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "D:\\farmland boundary\\deeplab_boundary.onnx"
}
```

---

### 📊 Model Information

```bash
GET /info
```

**Response:**
```json
{
  "model_type": "DeepLabV3+ ResNet34",
  "input_format": "ONNX",
  "input_shape": [1, 3, 768, 768],
  "output_shape": [1, 1, 768, 768],
  "task": "Farmland boundary segmentation",
  "model_path": "D:\\farmland boundary\\deeplab_boundary.onnx"
}
```

---

### 🔮 Single Image Prediction

```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- threshold: Confidence threshold 0-1 (default: 0.25)
- return_mask: Return base64 mask (default: false)
```

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/predict?threshold=0.25&return_mask=true" \
  -F "file=@image.jpg"
```

**Example with Python:**
```python
import requests

with open("satellite_tile.jpg", "rb") as f:
    files = {"file": f}
    params = {"threshold": 0.25, "return_mask": True}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files,
        params=params
    )
    
result = response.json()
geojson = result["geojson"]
metadata = result["metadata"]
mask = result.get("mask")  # Base64 encoded PNG
```

**Response:**
```json
{
  "geojson": {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "features": [
      {
        "type": "Feature",
        "id": 0,
        "properties": {
          "index": 0,
          "area": 390.50,
          "perimeter": 173.36,
          "bbox": {"x": 910, "y": 661, "width": 80, "height": 16},
          "image_dimensions": {"width": 1920, "height": 915}
        },
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[910.0, 662.0], [910.0, 665.0], ...]]
        }
      }
    ]
  },
  "metadata": {
    "image_width": 1920,
    "image_height": 915,
    "num_contours": 6,
    "threshold": 0.25,
    "filename": "satellite_tile.jpg"
  },
  "mask": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

---

### 📦 Batch Prediction

```bash
POST /predict-batch
Content-Type: multipart/form-data

Parameters:
- files: Multiple image files (required)
- threshold: Confidence threshold 0-1 (default: 0.25)
```

**Example with Python:**
```python
import requests
from pathlib import Path

# Get all JPG files
image_paths = list(Path("images/").glob("*.jpg"))

files = [("files", open(img, "rb")) for img in image_paths]
params = {"threshold": 0.25}

response = requests.post(
    "http://localhost:8000/predict-batch",
    files=files,
    params=params
)

results = response.json()["results"]

for result in results:
    if result["status"] == "success":
        print(f"{result['filename']}: {result['num_contours']} contours")
```

**Response:**
```json
{
  "results": [
    {
      "filename": "satellite_tile_1.jpg",
      "status": "success",
      "geojson": {...},
      "num_contours": 6,
      "dimensions": {"width": 1920, "height": 915}
    },
    {
      "filename": "satellite_tile_2.jpg",
      "status": "success",
      "geojson": {...},
      "num_contours": 8,
      "dimensions": {"width": 1920, "height": 915}
    }
  ]
}
```

---

## Configuration

Edit `.env` file to customize:

```env
# SERVER
HOST=0.0.0.0
PORT=8000
DEBUG=False

# MODEL
ONNX_MODEL_PATH=D:\farmland boundary\deeplab_boundary.onnx

# INFERENCE
DEFAULT_THRESHOLD=0.25
DEFAULT_MIN_AREA=100
INFERENCE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# CORS
CORS_ORIGINS=*

# LOGGING
LOG_LEVEL=INFO
```

---

## Testing

### Run Test Suite

```bash
python test_client.py
```

### Individual Tests

```bash
# Health check
python test_client.py health

# Model info
python test_client.py info

# Single image
python test_client.py predict "path/to/image.jpg" 0.25

# Batch
python test_client.py batch "path/to/images" 5
```

---

## File Structure

```
fastapi_server/
├── main.py                 # FastAPI application
├── inference.py            # ONNX inference engine
├── utils.py                # Contour & GeoJSON utilities
├── config.py               # Configuration management
├── test_client.py          # API testing client
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md              # This file
```

---

## Key Features

### 🎯 **Segmentation**
- DeepLabV3+ ResNet34 model
- ONNX format for cross-platform deployment
- Supports CPU and GPU inference

### 🔍 **Contour Detection**
- Automatic boundary extraction
- Morphological operations (closing, opening)
- Minimum area filtering
- Contour simplification (Ramer-Douglas-Peucker)

### 📍 **GeoJSON Output**
- Polygon coordinates in pixel space
- Support for geographic coordinate transformation
- Feature properties: area, perimeter, bounding box
- CRS information (EPSG:4326)

### 🖼️ **Image Processing**
- Batch processing support
- Configurable confidence threshold
- Optional mask image return (base64)
- Automatic resizing and normalization

---

## Performance

### Inference Speed
- Single image (1920×915): ~0.5-1s (CPU)
- Batch (10 images): ~5-10s (CPU)

### Model Info
- Input: 768×768 RGB image
- Output: 768×768 single-channel mask
- Weights: ~41 MB
- Framework: PyTorch (exported to ONNX)

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `503 Model not loaded` | Model file missing | Check ONNX_MODEL_PATH in .env |
| `400 Invalid image file` | Corrupted image | Verify image format (JPG, PNG) |
| `400 Bad argument` | Image decode failed | Try different image format |
| `Connection refused` | Server not running | Run `python main.py` |

---

## Integration Examples

### Chrome Extension
```javascript
// Send image to FastAPI server
const formData = new FormData();
formData.append('file', imageFile);
formData.append('threshold', 0.25);

fetch('http://localhost:8000/predict?return_mask=true', {
  method: 'POST',
  body: formData
})
.then(r => r.json())
.then(data => {
  // data.geojson contains polygons
  // data.mask contains base64 image
});
```

### Command Line
```bash
# Predict on single image
curl -F "file=@tile.jpg" http://localhost:8000/predict | jq '.geojson'

# Batch with threshold
for img in *.jpg; do
  curl -F "file=@$img" "http://localhost:8000/predict?threshold=0.3"
done
```

---

## Deployment

### Production Setup

```bash
# Use Gunicorn with Uvicorn workers
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t deeplab-api .
docker run -p 8000:8000 deeplab-api
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check Python version (requires 3.8+)
python --version

# Verify ONNX model exists
python -c "import os; print(os.path.exists(r'D:\farmland boundary\deeplab_boundary.onnx'))"

# Check port availability
netstat -an | findstr 8000
```

### Model Loading Fails
```bash
# Test ONNX model directly
python -c "
import onnxruntime as ort
session = ort.InferenceSession(r'D:\farmland boundary\deeplab_boundary.onnx')
print('Model loaded successfully')
"
```

### Slow Inference
- Use GPU provider (install CUDA-enabled onnxruntime)
- Reduce image size before sending
- Use batch endpoint for multiple images

---

## Next Steps

✅ **FastAPI Server**: Complete
- [x] ONNX inference
- [x] Single image prediction
- [x] Batch prediction
- [x] GeoJSON output
- [x] API documentation

📋 **Roadmap**:
1. **Chrome Extension**: Overlay GeoJSON on map
2. **Geographic Coordinates**: Transform pixel to lat/lon
3. **Database Storage**: Cache predictions
4. **Real-time Updates**: WebSocket support
5. **Model Management**: Version control & A/B testing

---

## Support

For issues or questions:
1. Check `/docs` endpoint for API details
2. Review test output: `python test_client.py`
3. Check server logs for error messages
4. Verify model file exists and is valid

---

**Version**: 1.0.0  
**Updated**: June 7, 2026  
**Framework**: FastAPI + ONNX Runtime
