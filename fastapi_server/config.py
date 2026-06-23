# ============================================================
# CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SERVER SETTINGS
# ============================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ============================================================
# MODEL SETTINGS
# ============================================================

ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    r"D:\farmland boundary\boundary_unetpp_b3_v2.onnx"
)

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.25))
DEFAULT_MIN_AREA = int(os.getenv("DEFAULT_MIN_AREA", 100))

# ============================================================
# INFERENCE SETTINGS
# ============================================================

INFERENCE_PROVIDERS = os.getenv(
    "INFERENCE_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider"
).split(",")

# ============================================================
# IMAGE SETTINGS
# ============================================================

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 4096))
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

# ============================================================
# CORS SETTINGS
# ============================================================

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*"
).split(",")

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

print(f"Configuration Loaded:")
print(f"  Host: {HOST}:{PORT}")
print(f"  Model: {ONNX_MODEL_PATH}")
print(f"  Threshold: {DEFAULT_THRESHOLD}")
print(f"  Providers: {INFERENCE_PROVIDERS}")
