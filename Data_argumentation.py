#Data Augmentation
import os
import cv2
import numpy as np

# ------------------------
# Paths
# ------------------------

IMAGE_DIR = r"D:\farmland boundary\dataset\images"
MASK_DIR = r"D:\farmland boundary\dataset\masks"
BOUNDARY_DIR = r"/Users/nirmal_25/Desktop/Files/Chrome_Extension_Ankit_Sir/Farmland_Segmentation_Code/farmland-segmentation/Train/boundary_mask"

# ------------------------
# Current numbering
# ------------------------

existing = [
    int(os.path.splitext(f)[0])
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".jpg")
]

next_id = max(existing) + 1

print("Starting from:", next_id)

# ------------------------
# Augmentation Functions
# ------------------------

def rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)

def rotate270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def hflip(img):
    return cv2.flip(img,1)

def vflip(img):
    return cv2.flip(img,0)

def brighter(img):
    return cv2.convertScaleAbs(img,alpha=1.2,beta=20)

def darker(img):
    return cv2.convertScaleAbs(img,alpha=0.8,beta=-15)

def contrast_high(img):
    return cv2.convertScaleAbs(img,alpha=1.4,beta=0)

def contrast_low(img):
    return cv2.convertScaleAbs(img,alpha=0.7,beta=0)

def blur(img):
    return cv2.GaussianBlur(img,(5,5),0)

def random_crop(img,mask,boundary):

    h,w=img.shape[:2]

    crop=0.85

    nh=int(h*crop)
    nw=int(w*crop)

    x=np.random.randint(0,w-nw)
    y=np.random.randint(0,h-nh)

    img=img[y:y+nh,x:x+nw]
    mask=mask[y:y+nh,x:x+nw]
    boundary=boundary[y:y+nh,x:x+nw]

    img=cv2.resize(img,(w,h))
    mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
    boundary=cv2.resize(boundary,(w,h),interpolation=cv2.INTER_NEAREST)

    return img,mask,boundary

# ------------------------
# Augmentation list
# ------------------------

augmentations = [
    ("hflip",hflip),
    ("vflip",vflip),
    ("rot90",rotate90),
    ("rot180",rotate180),
    ("rot270",rotate270),
    ("bright",brighter),
    ("dark",darker),
    ("contrast+",contrast_high),
    ("contrast-",contrast_low),
    ("blur",blur)
]

# ------------------------
# Process all images
# ------------------------

files=sorted(os.listdir(IMAGE_DIR))

for file in files:

    if not file.endswith(".jpg"):
        continue

    name=os.path.splitext(file)[0]

    image=cv2.imread(os.path.join(IMAGE_DIR,file))
    mask=cv2.imread(os.path.join(MASK_DIR,name+".png"),0)
    boundary=cv2.imread(os.path.join(BOUNDARY_DIR,name+".png"),0)

    # --------------------
    # Simple augmentations
    # --------------------

    for aug_name,aug in augmentations:

        img2=aug(image)
        mask2=aug(mask)
        boundary2=aug(boundary)

        cv2.imwrite(os.path.join(IMAGE_DIR,f"{next_id}.jpg"),img2)
        cv2.imwrite(os.path.join(MASK_DIR,f"{next_id}.png"),mask2)
        cv2.imwrite(os.path.join(BOUNDARY_DIR,f"{next_id}.png"),boundary2)

        next_id+=1

    # --------------------
    # Random Crop
    # --------------------

    img2,mask2,boundary2=random_crop(image,mask,boundary)

    cv2.imwrite(os.path.join(IMAGE_DIR,f"{next_id}.jpg"),img2)
    cv2.imwrite(os.path.join(MASK_DIR,f"{next_id}.png"),mask2)
    cv2.imwrite(os.path.join(BOUNDARY_DIR,f"{next_id}.png"),boundary2)

    next_id+=1

print("Done")
print("Total Images:",len(os.listdir(IMAGE_DIR)))