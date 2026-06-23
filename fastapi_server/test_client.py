# ============================================================
# TEST CLIENT FOR FASTAPI SERVER
# ============================================================

import requests
import cv2
import json
import base64
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://localhost:8000"
SAMPLE_IMAGES_PATH = r"D:\farmland boundary\New folder\images"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_server_health():
    """Check if server is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Server is HEALTHY")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("❌ Server returned non-200 status")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print(f"   Try: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================================
# GET MODEL INFO
# ============================================================

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/info")
        if response.status_code == 200:
            print("\n📊 MODEL INFORMATION:")
            print(json.dumps(response.json(), indent=2))
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_single_image(image_path, threshold=0.25, return_mask=False):
    """
    Predict on a single image
    
    Parameters:
    - image_path: Path to image file
    - threshold: Confidence threshold
    - return_mask: Whether to return base64 mask
    """
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        print(f"\n🔮 PREDICTING: {Path(image_path).name}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'threshold': threshold,
                'return_mask': return_mask
            }
            
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Prediction successful!")
            print(f"   Image: {data['metadata']['filename']}")
            print(f"   Dimensions: {data['metadata']['image_width']}x{data['metadata']['image_height']}")
            print(f"   Contours found: {data['metadata']['num_contours']}")
            print(f"   Threshold: {data['metadata']['threshold']}")
            
            # Show GeoJSON summary
            geojson = data['geojson']
            print(f"\n📍 GeoJSON FeatureCollection:")
            print(f"   Type: {geojson['type']}")
            print(f"   CRS: {geojson['crs']['properties']['name']}")
            print(f"   Features: {len(geojson['features'])}")
            
            if len(geojson['features']) > 0:
                print(f"\n   First polygon properties:")
                first_feature = geojson['features'][0]
                props = first_feature['properties']
                print(f"   - Area: {props['area']:.2f} pixels²")
                print(f"   - Perimeter: {props['perimeter']:.2f} pixels")
                print(f"   - BBox: {props['bbox']}")
                
                # Show first few coordinates
                coords = first_feature['geometry']['coordinates'][0]
                print(f"   - Polygon points: {len(coords)}")
                print(f"   - First 3 points: {coords[:3]}")
            
            # Save full GeoJSON
            output_path = Path(image_path).stem + "_predictions.geojson"
            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            print(f"\n💾 GeoJSON saved: {output_path}")
            
            # Save mask if returned
            if 'mask' in data:
                mask_b64 = data['mask']
                mask_data = base64.b64decode(mask_b64)
                mask_path = Path(image_path).stem + "_mask.png"
                with open(mask_path, 'wb') as f:
                    f.write(mask_data)
                print(f"💾 Mask saved: {mask_path}")
            
            return data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(image_dir, num_images=3, threshold=0.25):
    """
    Predict on multiple images
    
    Parameters:
    - image_dir: Directory containing images
    - num_images: Maximum number of images to process
    - threshold: Confidence threshold
    """
    
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"❌ Directory not found: {image_dir}")
        return None
    
    # Get image files
    image_files = list(image_path.glob("*.jpg")) + \
                  list(image_path.glob("*.png")) + \
                  list(image_path.glob("*.jpeg"))
    
    image_files = image_files[:num_images]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return None
    
    try:
        print(f"\n🔮 BATCH PREDICTING: {len(image_files)} images")
        
        files = [('files', open(img, 'rb')) for img in image_files]
        params = {'threshold': threshold}
        
        response = requests.post(
            f"{API_URL}/predict-batch",
            files=files,
            params=params
        )
        
        # Close files
        for _, f in files:
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            
            print(f"✅ Batch prediction complete!")
            
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] == 'error')
            
            print(f"   ✓ Successful: {successful}")
            print(f"   ✗ Failed: {failed}")
            
            # Save results
            output_path = "batch_predictions.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved: {output_path}")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            for result in results:
                filename = result['filename']
                if result['status'] == 'success':
                    num_contours = result['num_contours']
                    dims = result['dimensions']
                    print(f"   ✓ {filename}: {num_contours} contours ({dims['width']}x{dims['height']})")
                else:
                    print(f"   ✗ {filename}: {result['message']}")
            
            return data
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================
# MAIN TEST ROUTINE
# ============================================================

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("FastAPI UNet++ Boundary Prediction Client")
    print("=" * 60)
    
    # ============================================================
    # HEALTH CHECK
    # ============================================================
    
    if not check_server_health():
        print("\n⚠️  Server is not running!")
        print("Start the server with: python main.py")
        return
    
    # ============================================================
    # MODEL INFO
    # ============================================================
    
    get_model_info()
    
    # ============================================================
    # SINGLE IMAGE TEST
    # ============================================================
    
    sample_images = list(Path(SAMPLE_IMAGES_PATH).glob("*.jpg")) + \
                    list(Path(SAMPLE_IMAGES_PATH).glob("*.png"))
    
    if sample_images:
        print(f"\n{'=' * 60}")
        print("SINGLE IMAGE PREDICTION")
        print(f"{'=' * 60}")
        
        predict_single_image(
            str(sample_images[0]),
            threshold=0.25,
            return_mask=True
        )
    else:
        print(f"\n⚠️  No sample images found in {SAMPLE_IMAGES_PATH}")
    
    # ============================================================
    # BATCH PREDICTION TEST
    # ============================================================
    
    if sample_images:
        print(f"\n{'=' * 60}")
        print("BATCH PREDICTION")
        print(f"{'=' * 60}")
        
        predict_batch(
            SAMPLE_IMAGES_PATH,
            num_images=3,
            threshold=0.25
        )
    
    print(f"\n{'=' * 60}")
    print("✅ All tests complete!")
    print(f"{'=' * 60}")

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            check_server_health()
        elif command == "info":
            get_model_info()
        elif command == "predict" and len(sys.argv) > 2:
            image_path = sys.argv[2]
            threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
            predict_single_image(image_path, threshold, return_mask=True)
        elif command == "batch" and len(sys.argv) > 2:
            image_dir = sys.argv[2]
            num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            predict_batch(image_dir, num_images)
        else:
            print("Usage:")
            print("  python test_client.py health")
            print("  python test_client.py info")
            print("  python test_client.py predict <image_path> [threshold]")
            print("  python test_client.py batch <image_dir> [num_images]")
            print("  python test_client.py (run all tests)")
    else:
        main()
