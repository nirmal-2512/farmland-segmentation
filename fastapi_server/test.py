import cv2
import numpy as np

# Load your mask_binary.png
mask = cv2.imread("/Users/nirmal_25/Desktop/Files/Chrome_Extension_Ankit_Sir/Farmland_Segmentation_Code/farmland-segmentation/fastapi_server/debug_outputs/run_20260629_080936_753755/mask_binary.png", 0)

# Step 1 - after clean (just dilate directly, skip clean for now)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(mask, kernel, iterations=2)

# Step 2 - invert
inverted = cv2.bitwise_not(dilated)

# Step 3 - border mask
border_mask = np.zeros_like(inverted)
border_mask[1:-1, 1:-1] = 255
inverted = cv2.bitwise_and(inverted, border_mask)

# Step 4 - connected components
h, w = inverted.shape[:2]
max_area = h * w * 0.5
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

print(f"Total components: {num_labels}")
areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
areas.sort(reverse=True)
print(f"Top 10 component areas: {areas[:10]}")
print(f"min_area filter (500) would keep: {sum(1 for a in areas if 500 <= a <= max_area)}")
print(f"max_area threshold: {max_area:.0f}")

# Save intermediate images
cv2.imwrite("/tmp/dilated.png", dilated)
cv2.imwrite("/tmp/inverted.png", inverted)
print("Saved dilated.png and inverted.png to /tmp/")