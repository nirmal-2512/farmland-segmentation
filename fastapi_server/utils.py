# ============================================================
# UTILITY FUNCTIONS FOR CONTOUR & GEOJSON
# ============================================================

import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================
# MASK CLEANING AND POSTPROCESSING
# ============================================================

def clean_boundary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    closing_iterations: int = 2,
    opening_iterations: int = 1,
    min_area: int = 100
) -> np.ndarray:
    """
    Clean a binary boundary mask before contour extraction.

    Steps:
    - Morphological closing to connect broken edges
    - Morphological opening to remove noise
    - Connected components filtering to remove small blobs
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == idx] = 255

    return cleaned


def draw_contours_overlay(image: np.ndarray, contours: List[np.ndarray], color=(0, 0, 255), thickness=2) -> np.ndarray:
    """
    Draw contours on a copy of the original image for debug overlay.
    """
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay

def boundary_mask_to_field_regions(
    boundary_mask: np.ndarray,
    kernel_size: int = 3,         # was 13 — much smaller for thin boundaries
    closing_iterations: int = 2,  # was 3
    opening_iterations: int = 1,
    dilation_iterations: int = 2, # was 3
    min_area: int = 500
) -> np.ndarray:
    """
    Convert a boundary line mask into a field region mask.
    Tuned for thin boundary lines (< 5% white pixel coverage).
    """

    # Step 1 — Clean boundary lines
    cleaned = clean_boundary_mask(
        boundary_mask,
        kernel_size=kernel_size,
        closing_iterations=closing_iterations,
        opening_iterations=opening_iterations,
        min_area=min_area
    )

    # Step 2 — Dilate just enough to close small gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(cleaned, kernel, iterations=dilation_iterations)

    # Step 3 — Invert: boundaries become black, field interiors become white
    inverted = cv2.bitwise_not(dilated)

    # Step 4 — Remove image border regions
    border_mask = np.zeros_like(inverted)
    border_mask[1:-1, 1:-1] = 255
    inverted = cv2.bitwise_and(inverted, border_mask)

    # Step 5 — Connected component filtering
    h, w = inverted.shape[:2]
    max_area = h * w * 0.95

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    output = np.zeros_like(inverted)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            output[labels == label] = 255

    return output


# ============================================================
# CONTOUR EXTRACTION
# ============================================================

def extract_contours(mask, min_area=100):
    """
    Extract contours from binary mask
    
    Parameters:
    - mask: Binary mask image
    - min_area: Minimum area to filter small contours
    
    Returns:
    - List of contours with area > min_area
    """
    
    try:
        # ====================================================
        # FIND CONTOURS
        # ====================================================
        
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # ====================================================
        # FILTER BY AREA
        # ====================================================
        
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                filtered_contours.append(contour)
        
        logger.info(f"Extracted {len(filtered_contours)} contours")
        
        return filtered_contours
    
    except Exception as e:
        logger.error(f"Contour extraction error: {e}")
        return []

# ============================================================
# CONTOUR SIMPLIFICATION
# ============================================================

def simplify_contour(contour, epsilon_ratio=0.01):
    """
    Simplify contour using Ramer-Douglas-Peucker algorithm
    
    Parameters:
    - contour: OpenCV contour
    - epsilon_ratio: Simplification parameter (% of perimeter)
    
    Returns:
    - Simplified contour
    """
    
    try:
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter
        
        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )
        
        return simplified
    
    except Exception as e:
        logger.error(f"Contour simplification error: {e}")
        return contour

# ============================================================
# CONTOUR TO POLYGON COORDINATES
# ============================================================

def contour_to_polygon(contour):
    """
    Convert OpenCV contour to polygon coordinates
    
    Parameters:
    - contour: OpenCV contour
    
    Returns:
    - List of [lon, lat] coordinates (normalized to 0-1 in pixel space)
    """
    
    try:
        # Simplify contour
        simplified = simplify_contour(contour)
        
        # Extract coordinates
        coords = []
        for point in simplified:
            x, y = point[0]
            coords.append([float(x), float(y)])
        
        # Close polygon (add first point at end)
        if len(coords) > 0 and coords[0] != coords[-1]:
            coords.append(coords[0])
        
        return coords
    
    except Exception as e:
        logger.error(f"Contour to polygon error: {e}")
        return []

# ============================================================
# GEOJSON GENERATION
# ============================================================

def contours_to_geojson(
    contours: List,
    image_width: int = 768,
    image_height: int = 768,
    crs: str = "EPSG:4326"
) -> Dict[str, Any]:
    """
    Convert contours to GeoJSON FeatureCollection
    
    Parameters:
    - contours: List of OpenCV contours
    - image_width: Image width in pixels
    - image_height: Image height in pixels
    - crs: Coordinate reference system
    
    Returns:
    - GeoJSON FeatureCollection
    """
    
    features = []
    
    for idx, contour in enumerate(contours):
        try:
            # ================================================
            # GET CONTOUR PROPERTIES
            # ================================================
            
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # ================================================
            # GET BOUNDING BOX
            # ================================================
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # ================================================
            # CONVERT TO POLYGON
            # ================================================
            
            polygon_coords = contour_to_polygon(contour)
            
            if len(polygon_coords) < 3:
                continue
            
            # ================================================
            # NORMALIZE COORDINATES (optional - pixel space)
            # ================================================
            
            # If you need geographic coordinates, transform here
            # For now, keeping in pixel space
            
            # ================================================
            # CREATE FEATURE
            # ================================================
            
            feature = {
                "type": "Feature",
                "id": idx,
                "properties": {
                    "index": idx,
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "image_dimensions": {
                        "width": image_width,
                        "height": image_height
                    }
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                }
            }
            
            features.append(feature)
        
        except Exception as e:
            logger.error(f"Feature creation error for contour {idx}: {e}")
            continue
    
    # ================================================================
    # CREATE FEATURECOLLECTION
    # ================================================================
    
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": crs
            }
        },
        "features": features
    }
    
    return geojson

# ============================================================
# PIXEL TO GEOGRAPHIC COORDINATES
# ============================================================

def pixel_to_geo(
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    bounds: Dict[str, float]
) -> tuple:
    """
    Convert pixel coordinates to geographic coordinates
    using Web Mercator projection to match Google Maps exactly.
    """
    try:
        # Longitude is linear — safe to interpolate directly
        norm_x = pixel_x / image_width
        lon = bounds["west"] + norm_x * (bounds["east"] - bounds["west"])

        # Latitude is NOT linear on Mercator maps
        # Must convert bounds to Mercator Y, interpolate, then convert back

        def lat_to_mercator_y(lat_deg):
            lat_rad = np.radians(lat_deg)
            return np.log(np.tan(np.pi / 4 + lat_rad / 2))

        def mercator_y_to_lat(y):
            return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

        north_y = lat_to_mercator_y(bounds["north"])
        south_y = lat_to_mercator_y(bounds["south"])

        norm_y = pixel_y / image_height
        mercator_y = north_y - norm_y * (north_y - south_y)
        lat = mercator_y_to_lat(mercator_y)

        return (lon, lat)

    except Exception as e:
        logger.error(f"Pixel to geo conversion error: {e}")
        return (0.0, 0.0)

# ============================================================
# GEOGRAPHIC DISTANCE AND AREA HELPERS
# ============================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two points."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def polygon_area_geo(coords: List[List[float]]) -> float:
    """Approximate polygon area in square meters using spherical coordinates."""
    if len(coords) < 3:
        return 0.0

    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += np.radians(lon2 - lon1) * (
            np.sin(np.radians(lat1)) + np.sin(np.radians(lat2))
        )

    return abs(total) * (6371000.0 ** 2) / 2.0


def polygon_perimeter_geo(coords: List[List[float]]) -> float:
    """Approximate perimeter in meters for geographic polygon coordinates."""
    if len(coords) < 2:
        return 0.0

    perimeter = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        perimeter += haversine_distance(lat1, lon1, lat2, lon2)

    return perimeter

# ============================================================
# GEOJSON WITH GEOGRAPHIC COORDINATES
# ============================================================

def contours_to_geojson_geographic(
    contours: List,
    image_width: int,
    image_height: int,
    bounds: Dict[str, float],
    crs: str = "EPSG:4326"
) -> Dict[str, Any]:
    """
    Convert contours to GeoJSON with geographic coordinates
    
    Parameters:
    - contours: List of OpenCV contours
    - image_width, image_height: Image dimensions
    - bounds: Geographic bounds (north, south, east, west)
    - crs: Coordinate reference system
    
    Returns:
    - GeoJSON FeatureCollection with geographic coordinates
    """
    
    features = []
    
    for idx, contour in enumerate(contours):
        try:
            # ================================================
            # CONVERT TO POLYGON
            # ================================================
            
            polygon_coords = contour_to_polygon(contour)
            
            if len(polygon_coords) < 3:
                continue
            
            # ================================================
            # CONVERT TO GEOGRAPHIC COORDINATES
            # ================================================
            
            geo_coords = []
            for pixel_x, pixel_y in polygon_coords:
                lon, lat = pixel_to_geo(
                    pixel_x,
                    pixel_y,
                    image_width,
                    image_height,
                    bounds
                )
                geo_coords.append([lon, lat])
            
            # ================================================
            # CREATE FEATURE
            # ================================================
            
            area = polygon_area_geo(geo_coords)
            perimeter = polygon_perimeter_geo(geo_coords)
            
            feature = {
                "type": "Feature",
                "id": idx,
                "properties": {
                    "index": idx,
                    "area_m2": float(area),
                    "perimeter_m": float(perimeter)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                }
            }
            
            features.append(feature)
        
        except Exception as e:
            logger.error(f"Geographic feature creation error: {e}")
            continue
    
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": crs
            }
        },
        "features": features
    }
    
    return geojson
