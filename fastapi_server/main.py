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

import config
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

ONNX_PATH = config.ONNX_MODEL_PATH

try:
    inference = ONNXInference(ONNX_PATH, providers=config.INFERENCE_PROVIDERS)
    logger.info("ONNX Model Loaded Successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    inference = None


# ============================================================
# DEBUG HELPERS
# ============================================================

def _create_debug_dir() -> Path:
    """Create a timestamped debug output directory for each request"""
    base_dir = Path(__file__).resolve().parent
    debug_root = base_dir / "debug_outputs"
    debug_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = debug_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_debug_images(
    run_dir: Path,
    original: np.ndarray,
    raw_mask: np.ndarray,
    mask_binary: np.ndarray,
    field_regions: np.ndarray
):
    """
    Save all 4 intermediate images to the debug folder.

    - original.png      : The raw image received from Chrome Extension
    - raw_mask.png      : Pure float output from ML model (scaled to 0-255)
    - mask_binary.png   : After applying threshold (input to boundary_mask_to_field_regions)
    - field_regions.png : Output of boundary_mask_to_field_regions (input to GeoJSON)
    """

    # 1. Original image sent to model
    cv2.imwrite(str(run_dir / "original.png"), original)
    logger.info(f"  Saved: original.png")

    # 2. Raw model output — scale float (0.0-1.0) to uint8 (0-255)
    # Brighter pixels = model is more confident there is a boundary
    raw_visual = (raw_mask * 255).astype(np.uint8)
    cv2.imwrite(str(run_dir / "raw_mask.png"), raw_visual)
    logger.info(f"  Saved: raw_mask.png  "
                f"[min={raw_mask.min():.3f} max={raw_mask.max():.3f} "
                f"mean={raw_mask.mean():.3f}]")

    # 3. Binary mask after threshold — white=boundary, black=background
    cv2.imwrite(str(run_dir / "mask_binary.png"), mask_binary)
    logger.info(f"  Saved: mask_binary.png")

    # 4. Field regions — output of boundary_mask_to_field_regions
    # This is what gets converted to GeoJSON contours
    cv2.imwrite(str(run_dir / "field_regions.png"), field_regions)
    logger.info(f"  Saved: field_regions.png")

    logger.info(f"  Debug folder: {run_dir}")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """Check server health and model availability"""
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
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        h, w = image.shape[:2]
        logger.info(f"Processing image: {w}x{h}")

        mask, debug_info = inference.predict(image)

        logger.info(f"Prediction shape: {mask.shape}")
        logger.info(f"Prediction stats: min={mask.min():.4f}, "
                    f"max={mask.max():.4f}, mean={mask.mean():.4f}")

        mask_binary = (mask > threshold).astype(np.uint8) * 255
        field_mask = boundary_mask_to_field_regions(mask_binary)
        contours = extract_contours(field_mask)

        logger.info(f"Found {len(contours)} contours")

        geojson_data = contours_to_geojson(
            contours, image_width=w, image_height=h
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

        if return_mask:
            import base64
            _, buffer = cv2.imencode('.png', field_mask)
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
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    try:
        for file in files:
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

            mask, _ = inference.predict(image)
            mask_binary = (mask > threshold).astype(np.uint8) * 255
            mask_clean = clean_boundary_mask(mask_binary)
            contours = extract_contours(mask_clean)
            geojson_data = contours_to_geojson(
                contours, image_width=w, image_height=h
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
    Always saves 4 debug images per request to debug_outputs/ folder.
    """

    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
        # ── Read image ───────────────────────────────────────
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        h, w = image.shape[:2]
        logger.info(f"Processing image: {w}x{h}")

        # ── Create debug folder (always) ─────────────────────
        run_dir = _create_debug_dir()
        logger.info(f"Debug folder created: {run_dir}")

        # ── Run model ────────────────────────────────────────
        mask, debug_info = inference.predict(image)

        logger.info(f"Prediction shape: {mask.shape}")
        logger.info(f"Prediction stats: min={mask.min():.4f}, "
                    f"max={mask.max():.4f}, mean={mask.mean():.4f}")

        # ── Apply threshold ──────────────────────────────────
        mask_binary = (mask > threshold).astype(np.uint8) * 255

        # ── Extract field regions ────────────────────────────
        field_mask = boundary_mask_to_field_regions(mask_binary)

        # ── Save all 4 debug images ──────────────────────────
        _save_debug_images(
            run_dir=run_dir,
            original=image,
            raw_mask=mask,
            mask_binary=mask_binary,
            field_regions=field_mask
        )

        # ── Save threshold variants if debug=True ────────────
        # Only when explicitly requested — saves extra images
        # at different thresholds to help tune the value
        if debug:
            for t in [0.35, 0.45, 0.50, 0.60]:
                t_mask = (mask > t).astype(np.uint8) * 255
                cv2.imwrite(
                    str(run_dir / f"threshold_{int(t * 100)}.png"),
                    t_mask
                )
            logger.info("  Saved threshold variants (debug=True)")

        # ── Extract contours ─────────────────────────────────
        contours = extract_contours(field_mask)
        logger.info(f"Found {len(contours)} contours")

        # ── Draw overlay and save ────────────────────────────
        overlay = draw_contours_overlay(
            image, contours, color=(0, 0, 255), thickness=2
        )
        cv2.imwrite(str(run_dir / "contours_overlay.png"), overlay)
        logger.info("  Saved: contours_overlay.png")

        # ── Convert to GeoJSON ───────────────────────────────
        geojson_data = contours_to_geojson_geographic(
            contours,
            image_width=image_width,
            image_height=image_height,
            bounds=bounds
        )

        # ── Build response ───────────────────────────────────
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
                "filename": file.filename,
                "debug_folder": str(run_dir)
            }
        }

        if return_mask:
            import base64
            _, buffer = cv2.imencode('.png', field_mask)
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
    """Get model information"""
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower()
    )