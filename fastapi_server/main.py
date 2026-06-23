# ============================================================
# FASTAPI SERVER FOR UNET++ BOUNDARY PREDICTION
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
import logging
import os
from pathlib import Path
from datetime import datetime

from inference import ONNXInference
from utils import (
    clean_boundary_mask,
    boundary_mask_to_field_regions,
    draw_contours_overlay,
    extract_contours,
    contours_to_geojson,
    contours_to_geojson_geographic
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="UNet++ Boundary Prediction API",
    description="Satellite tile boundary segmentation and GeoJSON conversion",
    version="1.0.0"
)

# ============================================================
# CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# INITIALIZE MODEL
# ============================================================

# Use the ONNX model included in the workspace. The path previously
# omitted the "New folder" component; update to the correct absolute path.
ONNX_PATH = r"D:\farmland boundary\New folder\boundary_unetpp_b3_v2.onnx"

try:
    inference = ONNXInference(ONNX_PATH)
    logger.info("ONNX Model Loaded Successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    inference = None


def _create_debug_dir():
    base_dir = Path(__file__).resolve().parent
    debug_root = base_dir / "debug_outputs"
    debug_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = debug_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """
    Check server health and model availability
    """
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": ONNX_PATH
    }

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.25,
    return_mask: bool = False
):
    """
    Predict boundary mask and extract contours as GeoJSON
    
    Parameters:
    - file: Satellite tile image (JPG, PNG)
    - threshold: Confidence threshold for mask (0-1)
    - return_mask: Whether to return base64 encoded mask image
    
    Returns:
    - geojson: GeoJSON FeatureCollection of boundary polygons
    - metadata: Image dimensions and processing info
    - mask (optional): Base64 encoded mask image
    """
    
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # ====================================================
        # READ IMAGE
        # ====================================================
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image file")
        
        h, w = image.shape[:2]
        
        logger.info(f"Processing image: {w}x{h}")
        
        # ====================================================
        # INFERENCE
        # ====================================================
        
        mask, debug_info = inference.predict(image)
        
        logger.info(f"Prediction shape: {mask.shape}")
        logger.info(f"Prediction stats: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")
        
        # ====================================================
        # THRESHOLD
        # ====================================================
        
        mask_binary = (mask > threshold).astype(np.uint8) * 255
        
        # ====================================================
        # FIELD REGION EXTRACTION FROM BOUNDARIES
        # ====================================================
        
        field_mask = boundary_mask_to_field_regions(mask_binary)
        
        # ====================================================
        # EXTRACT CONTOURS
        # ====================================================
        
        contours = extract_contours(field_mask)
        
        logger.info(f"Found {len(contours)} contours")
        
        # ====================================================
        # CONVERT TO GEOJSON
        # ====================================================
        
        geojson_data = contours_to_geojson(
            contours,
            image_width=w,
            image_height=h
        )
        
        response_data = {
            "geojson": geojson_data,
            "metadata": {
                "image_width": w,
                "image_height": h,
                "num_contours": len(contours),
                "threshold": threshold,
                "filename": file.filename
            }
        }
        
        # ====================================================
        # OPTIONAL: RETURN MASK
        # ====================================================
        
        if return_mask:
            _, buffer = cv2.imencode('.png', mask_clean)
            import base64
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            response_data["mask"] = mask_b64
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = 0.25
):
    """
    Predict on multiple images
    
    Parameters:
    - files: List of satellite tile images
    - threshold: Confidence threshold for mask
    
    Returns:
    - results: List of prediction results for each image
    """
    
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    results = []
    
    try:
        for file in files:
            # Read file contents
            contents = await file.read()
            
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Invalid image"
                })
                continue
            
            h, w = image.shape[:2]
            
            # PREDICT
            mask, _ = inference.predict(image)
            mask_binary = (mask > threshold).astype(np.uint8) * 255
            
            # MORPHOLOGICAL OPERATIONS
            mask_clean = clean_boundary_mask(mask_binary)
            
            # EXTRACT CONTOURS
            contours = extract_contours(mask_clean)
            
            # CONVERT TO GEOJSON
            geojson_data = contours_to_geojson(
                contours,
                image_width=w,
                image_height=h
            )
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "geojson": geojson_data,
                "num_contours": len(contours),
                "dimensions": {"width": w, "height": h}
            })
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ============================================================
# GEOREFERENCED PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-georef")
async def predict_georef(
    file: UploadFile = File(...),
    north: float = Form(...),
    south: float = Form(...),
    east: float = Form(...),
    west: float = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),
    threshold: float = Form(0.25),
    debug: bool = Form(False),
    return_mask: bool = Form(False)
):
    """
    Predict boundaries and georeference polygons using map bounds.
    """
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    if north <= south or east <= west:
        raise HTTPException(
            status_code=400,
            detail="Invalid map bounds provided"
        )

    bounds = {
        "north": north,
        "south": south,
        "east": east,
        "west": west
    }

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        h, w = image.shape[:2]

        debug_output_dir = _create_debug_dir() if debug else None
        if debug_output_dir is not None:
            _ = cv2.imwrite(str(debug_output_dir / 'debug_capture.png'), image)

        mask, debug_info = inference.predict(image, debug_output_dir=debug_output_dir)

        logger.info(f"Prediction shape: {mask.shape}")
        logger.info(f"Prediction stats: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")

        if debug_output_dir is not None:
            for t in [0.35, 0.45, 0.50, 0.60]:
                threshold_mask = (mask > t).astype(np.uint8) * 255
                cv2.imwrite(str(debug_output_dir / f'debug_threshold_{int(t*100)}.png'), threshold_mask)

        mask_binary = (mask > threshold).astype(np.uint8) * 255
        field_mask = boundary_mask_to_field_regions(mask_binary)

        if debug_output_dir is not None:
            cv2.imwrite(str(debug_output_dir / 'debug_boundary_cleaned.png'), field_mask)

        contours = extract_contours(field_mask)
        if debug_output_dir is not None:
            overlay = draw_contours_overlay(image, contours, color=(0, 0, 255), thickness=2)
            cv2.imwrite(str(debug_output_dir / 'debug_geojson_overlay.png'), overlay)

        geojson_data = contours_to_geojson_geographic(
            contours,
            image_width=image_width,
            image_height=image_height,
            bounds=bounds
        )

        response_data = {
            "geojson": geojson_data,
            "metadata": {
                "image_width": w,
                "image_height": h,
                "map_width": image_width,
                "map_height": image_height,
                "bounds": bounds,
                "num_contours": len(contours),
                "threshold": threshold,
                "filename": file.filename
            }
        }

        if return_mask:
            _, buffer = cv2.imencode('.png', mask_clean)
            import base64
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            response_data["mask"] = mask_b64

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Georeference prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/info")
async def model_info():
    """
    Get model information
    """
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "UNet++ EfficientNet-B3",
        "input_format": "ONNX",
        "input_shape": [1, 3, 768, 768],
        "output_shape": [1, 1, 768, 768],
        "task": "Farmland boundary segmentation",
        "model_path": ONNX_PATH
    }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
