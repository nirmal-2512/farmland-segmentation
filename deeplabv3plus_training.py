import os
import cv2
import sys
import math
import glob
import time
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# ----------------------------- USER PATHS -----------------------------
ROOT = r"D:\farmland boundary\New folder"
TRAIN1_IMAGES = os.path.join(ROOT, "Train", "farmlands")
TRAIN2_IMAGES = r"D:\farmland boundary\New folder\Train 2\images"
TRAIN1_XML = os.path.join(ROOT, "Train", "boundary mask", "Train1_annotations.xml")
TRAIN2_XML = os.path.join(r"D:\farmland boundary\New folder\Train 2", "boundarydet", "Train2_annotations.xml")
MASK_FOLDER = r"D:\farmland boundary\all_boundary_masks"
PREDICT_IMAGES = os.path.join(ROOT, "images")
PREDICTION_FOLDER = r"D:\farmland boundary\deeplab_predictions"
OVERLAY_FOLDER = r"D:\farmland boundary\overlay_predictions"
MODEL_PATH = r"D:\farmland boundary\deeplabv3plus_boundary_resnet101_best.pth"

os.makedirs(PREDICTION_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)

# ----------------------------- SETTINGS -----------------------------
IMAGE_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-4
NUM_WORKERS = 4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------- DEVICE -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device ->", DEVICE)
if DEVICE == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# ----------------------------- UTIL -----------------------------
def normalize_name(name):
    name = os.path.splitext(name)[0]
    name = name.lower().replace("_pred", "").replace("- copy", "").replace(" ", "").strip()
    return name


def parse_xml_to_masks(xml_path, images_dirs):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    candidates = root.findall(".//image") or root.findall(".//annotation") or [root]
    for node in candidates:
        filename = None
        for tag in ("filename","file","imageName","name"):
            el = node.find(tag)
            if el is not None and el.text:
                filename = el.text.strip()
                break
        if filename is None:
            filename = node.attrib.get("name") or node.attrib.get("filename")
        if filename is None:
            continue
        width = height = None
        size = node.find("size")
        if size is not None:
            w = size.find("width")
            h = size.find("height")
            if w is not None and h is not None:
                try:
                    width = int(w.text); height = int(h.text)
                except:
                    pass
        img_path = None
        if width is None or height is None:
            for d in images_dirs:
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    img_path = candidate
                    img = cv2.imread(candidate)
                    if img is not None:
                        height, width = img.shape[:2]
                    break
        if width is None or height is None:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        objects = node.findall(".//object") or node.findall(".//polygon") or []
        if not objects:
            objects = node.findall(".//polygon") or []
        for obj in objects:
            pts = []
            for pt in obj.findall(".//pt"):
                x = pt.find("x"); y = pt.find("y")
                if x is None or y is None:
                    continue
                try:
                    pts.append([int(float(x.text)), int(float(y.text))])
                except:
                    pass
            if not pts:
                pts_text = obj.find(".//points")
                if pts_text is not None and pts_text.text:
                    coord_pairs = pts_text.text.strip().split()
                    for cp in coord_pairs:
                        if ',' in cp:
                            a,b = cp.split(',')
                            try:
                                pts.append([int(float(a)), int(float(b))])
                            except:
                                pass
            if not pts:
                bnd = obj.find("bndbox")
                if bnd is not None:
                    xmin = bnd.find("xmin"); ymin = bnd.find("ymin")
                    xmax = bnd.find("xmax"); ymax = bnd.find("ymax")
                    try:
                        xmin = int(float(xmin.text)); ymin = int(float(ymin.text))
                        xmax = int(float(xmax.text)); ymax = int(float(ymax.text))
                        pts = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
                    except:
                        pts = []
            if pts:
                pts_np = np.array([pts], dtype=np.int32)
                cv2.fillPoly(mask, pts_np, 255)
        key = normalize_name(filename)
        result[key] = mask
    return result


class FarmlandDataset(Dataset):
    def __init__(self, image_dirs, mask_folder=None, xml_paths=None, transforms=None, img_size=IMAGE_SIZE):
        self.image_paths = []
        for d in image_dirs:
            if not os.path.isdir(d):
                continue
            self.image_paths += sorted(glob(os.path.join(d, "*.jpg")) + glob(os.path.join(d,"*.png")) + glob(os.path.join(d,"*.jpeg")))
        self.image_paths = sorted(list(set(self.image_paths)))
        self.mask_dict = {}
        if mask_folder and os.path.isdir(mask_folder):
            for m in glob(os.path.join(mask_folder, "*.png")) + glob(os.path.join(mask_folder, "*.jpg")):
                key = normalize_name(os.path.basename(m))
                self.mask_dict[key] = m
        if xml_paths:
            for xp in xml_paths:
                if os.path.exists(xp):
                    xml_masks = parse_xml_to_masks(xp, image_dirs)
                    for k,mask_arr in xml_masks.items():
                        self.mask_dict[k] = mask_arr
        self.samples = []
        for p in self.image_paths:
            key = normalize_name(os.path.basename(p))
            if key in self.mask_dict:
                self.samples.append((p, self.mask_dict[key]))
        print(f"Found {len(self.image_paths)} images, {len(self.samples)} usable (image+mask pairs).")
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_src = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(mask_src, str):
            mask = cv2.imread(mask_src, 0)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = mask_src.copy()
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype("float32")
        mask = np.expand_dims(mask, axis=0)
        return img, mask, img_path


train_transforms = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.2),
    ], p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.2),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

valid_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

image_dirs = [TRAIN1_IMAGES, TRAIN2_IMAGES]
xml_paths = []
if os.path.exists(TRAIN1_XML):
    xml_paths.append(TRAIN1_XML)
if os.path.exists(TRAIN2_XML):
    xml_paths.append(TRAIN2_XML)

full_dataset = FarmlandDataset(image_dirs=image_dirs, mask_folder=MASK_FOLDER, xml_paths=xml_paths, transforms=None, img_size=IMAGE_SIZE)
if len(full_dataset) < 1:
    raise RuntimeError("No training samples found. Check image/mask/XML paths.")

val_pct = 0.2
val_size = max(1, int(len(full_dataset) * val_pct))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

class WrappedDataset(Dataset):
    def __init__(self, base_ds, transforms):
        self.base = base_ds
        self.transforms = transforms
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, mask, img_path = self.base[idx]
        augmented = self.transforms(image=img, mask=mask[0]*255)
        img_t = augmented['image']
        mask_t = augmented['mask']
        mask_t = (mask_t > 127).astype("float32")
        mask_t = np.expand_dims(mask_t, axis=0)
        return img_t, mask_t, img_path

train_dataset = WrappedDataset(train_ds, train_transforms)
val_dataset = WrappedDataset(val_ds, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Loading DeepLabV3+ (resnet101 backbone)")
model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(DEVICE)

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")

def combined_loss(outputs, targets, bce_weight=0.5):
    bce = bce_loss(outputs, targets)
    probs = torch.sigmoid(outputs)
    d_loss = dice_loss(probs, targets)
    return bce * bce_weight + d_loss * (1 - bce_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


def calc_metrics_numpy(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = 1.0 if union == 0 else intersection / union
    total = y_true.sum() + y_pred.sum()
    dice = 1.0 if total == 0 else (2 * intersection) / total
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return iou, dice, precision, recall

best_val_dice = -1
history = {'train_loss':[], 'val_loss':[], 'val_dice':[], 'val_iou':[]}

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train", leave=False)
    for imgs, masks, _ in pbar:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        masks = masks.to(DEVICE, dtype=torch.float)
        preds = model(imgs)
        loss = combined_loss(preds, masks, bce_weight=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())
    avg_train_loss = running_loss / len(train_loader.dataset)
    model.eval()
    val_loss = 0.0
    dices = []
    ious = []
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)
            preds = model(imgs)
            loss = combined_loss(preds, masks, bce_weight=0.5)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(preds).cpu().numpy()
            gt = masks.cpu().numpy()
            for i in range(probs.shape[0]):
                p = (probs[i,0] > 0.5).astype(np.uint8)
                g = (gt[i,0] > 0.5).astype(np.uint8)
                iou, dice, prec, rec = calc_metrics_numpy(g, p)
                dices.append(dice); ious.append(iou)
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_dice = np.mean(dices) if dices else 0.0
    avg_val_iou = np.mean(ious) if ious else 0.0
    scheduler.step(avg_val_loss)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)
    print(f"Epoch {epoch}/{EPOCHS} TrainLoss: {avg_train_loss:.4f} ValLoss: {avg_val_loss:.4f} ValDice: {avg_val_dice:.4f} ValIoU: {avg_val_iou:.4f}")
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': best_val_dice,
        }, MODEL_PATH)
        print(f"Saved best model (ValDice={best_val_dice:.4f}) to {MODEL_PATH}")

plt.figure(figsize=(10,5))
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title("DeepLabV3+ Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("Loading best model for prediction...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

predict_images = sorted(glob(os.path.join(PREDICT_IMAGES, "*.jpg")) + glob(os.path.join(PREDICT_IMAGES, "*.png")) + glob(os.path.join(PREDICT_IMAGES, "*.jpeg")))
print("Total images to predict:", len(predict_images))

transform_predict = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

with torch.no_grad():
    for img_path in tqdm(predict_images, desc="Predicting"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig = img_bgr.copy()
        h,w = orig.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = transform_predict(image=img_rgb)
        tensor = aug['image'].unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_resized = cv2.resize(mask, (w,h))
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = base + "_pred.png"
        cv2.imwrite(os.path.join(PREDICTION_FOLDER, mask_name), mask_resized)
        overlay = orig.copy()
        overlay[mask_resized>0] = (0,0,255)
        cv2.imwrite(os.path.join(OVERLAY_FOLDER, mask_name), overlay)

print("Predictions saved to", PREDICTION_FOLDER, "and", OVERLAY_FOLDER)

# Metrics evaluation
gt_paths = glob(os.path.join(MASK_FOLDER, "*.png")) + glob(os.path.join(MASK_FOLDER, "*.jpg"))
pred_paths = glob(os.path.join(PREDICTION_FOLDER, "*_pred.png"))
pred_dict = { normalize_name(os.path.basename(p)): p for p in pred_paths }
ious = []; dices = []; precisions = []; recalls = []
matched = 0
for gt_path in gt_paths:
    key = normalize_name(os.path.basename(gt_path))
    if key not in pred_dict:
        continue
    pred_path = pred_dict[key]
    matched += 1
    gt = cv2.imread(gt_path, 0)
    pred = cv2.imread(pred_path, 0)
    if gt is None or pred is None:
        continue
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    iou, dice, prec, rec = calc_metrics_numpy(gt_bin, pred_bin)
    ious.append(iou); dices.append(dice); precisions.append(prec); recalls.append(rec)

print("Matched GT/pred samples:", matched)
avg_iou = np.mean(ious) if ious else 0.0
avg_dice = np.mean(dices) if dices else 0.0
avg_prec = np.mean(precisions) if precisions else 0.0
avg_rec = np.mean(recalls) if recalls else 0.0

print(f"IoU: {avg_iou:.4f}")
print(f"Dice: {avg_dice:.4f}")
print(f"Precision: {avg_prec:.4f}")
print(f"Recall: {avg_rec:.4f}")

metrics = ["IoU","Dice","Precision","Recall"]
values = [avg_iou, avg_dice, avg_prec, avg_rec]
plt.figure(figsize=(7,4))
bars = plt.bar(metrics, values, color=["#4C72B0","#55A868","#C44E52","#8172B3"])
plt.ylim(0,1)
for bar,v in zip(bars,values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha='center')
plt.title("Segmentation Metrics")
plt.show()

print("Pipeline complete.")
# deep_labv3plus_training.py
import os
import cv2
import sys
import math
import glob
import time
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# ----------------------------- USER PATHS -----------------------------
ROOT = r"D:\farmland boundary\New folder"
TRAIN1_IMAGES = os.path.join(ROOT, "Train", "farmlands")
TRAIN2_IMAGES = r"D:\farmland boundary\New folder\Train 2\images"
TRAIN1_XML = os.path.join(ROOT, "Train", "boundary mask", "Train1_annotations.xml")
TRAIN2_XML = os.path.join(r"D:\farmland boundary\New folder\Train 2", "boundarydet", "Train2_annotations.xml")
MASK_FOLDER = r"D:\farmland boundary\all_boundary_masks"
PREDICT_IMAGES = os.path.join(ROOT, "images")
PREDICTION_FOLDER = r"D:\farmland boundary\deeplab_predictions"
OVERLAY_FOLDER = r"D:\farmland boundary\overlay_predictions"
MODEL_PATH = r"D:\farmland boundary\deeplabv3plus_boundary_resnet101_best.pth"

os.makedirs(PREDICTION_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)

# ----------------------------- SETTINGS -----------------------------
IMAGE_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-4
NUM_WORKERS = 4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------- DEVICE -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device ->", DEVICE)
if DEVICE == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# ----------------------------- UTIL -----------------------------
def normalize_name(name):
    name = os.path.splitext(name)[0]
    name = name.lower().replace("_pred", "").replace("- copy", "").replace(" ", "").strip()
    return name


def parse_xml_to_masks(xml_path, images_dirs):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    candidates = root.findall(".//image") or root.findall(".//annotation") or [root]
    for node in candidates:
        filename = None
        for tag in ("filename","file","imageName","name"):
            el = node.find(tag)
            if el is not None and el.text:
                filename = el.text.strip()
                break
        if filename is None:
            filename = node.attrib.get("name") or node.attrib.get("filename")
        if filename is None:
            continue
        width = height = None
        size = node.find("size")
        if size is not None:
            w = size.find("width")
            h = size.find("height")
            if w is not None and h is not None:
                try:
                    width = int(w.text); height = int(h.text)
                except:
                    pass
        img_path = None
        if width is None or height is None:
            for d in images_dirs:
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    img_path = candidate
                    img = cv2.imread(candidate)
                    if img is not None:
                        height, width = img.shape[:2]
                    break
        if width is None or height is None:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        objects = node.findall(".//object") or node.findall(".//polygon") or []
        if not objects:
            objects = node.findall(".//polygon") or []
        for obj in objects:
            pts = []
            for pt in obj.findall(".//pt"):
                x = pt.find("x"); y = pt.find("y")
                if x is None or y is None:
                    continue
                try:
                    pts.append([int(float(x.text)), int(float(y.text))])
                except:
                    pass
            if not pts:
                pts_text = obj.find(".//points")
                if pts_text is not None and pts_text.text:
                    coord_pairs = pts_text.text.strip().split()
                    for cp in coord_pairs:
                        if ',' in cp:
                            a,b = cp.split(',')
                            try:
                                pts.append([int(float(a)), int(float(b))])
                            except:
                                pass
            if not pts:
                bnd = obj.find("bndbox")
                if bnd is not None:
                    xmin = bnd.find("xmin"); ymin = bnd.find("ymin")
                    xmax = bnd.find("xmax"); ymax = bnd.find("ymax")
                    try:
                        xmin = int(float(xmin.text)); ymin = int(float(ymin.text))
                        xmax = int(float(xmax.text)); ymax = int(float(ymax.text))
                        pts = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
                    except:
                        pts = []
            if pts:
                pts_np = np.array([pts], dtype=np.int32)
                cv2.fillPoly(mask, pts_np, 255)
        key = normalize_name(filename)
        result[key] = mask
    return result

# ----------------------------- DATASET -----------------------------
class FarmlandDataset(Dataset):
    def __init__(self, image_dirs, mask_folder=None, xml_paths=None, transforms=None, img_size=IMAGE_SIZE):
        self.image_paths = []
        for d in image_dirs:
            if not os.path.isdir(d):
                continue
            self.image_paths += sorted(glob(os.path.join(d, "*.jpg")) + glob(os.path.join(d,"*.png")) + glob(os.path.join(d,"*.jpeg")))
        self.image_paths = sorted(list(set(self.image_paths)))
        self.mask_dict = {}
        if mask_folder and os.path.isdir(mask_folder):
            for m in glob(os.path.join(mask_folder, "*.png")) + glob(os.path.join(mask_folder, "*.jpg")):
                key = normalize_name(os.path.basename(m))
                self.mask_dict[key] = m
        if xml_paths:
            for xp in xml_paths:
                if os.path.exists(xp):
                    xml_masks = parse_xml_to_masks(xp, image_dirs)
                    for k,mask_arr in xml_masks.items():
                        self.mask_dict[k] = mask_arr
        self.samples = []
        for p in self.image_paths:
            key = normalize_name(os.path.basename(p))
            if key in self.mask_dict:
                self.samples.append((p, self.mask_dict[key]))
        print(f"Found {len(self.image_paths)} images, {len(self.samples)} usable (image+mask pairs).")
        self.transforms = transforms
        self.img_size = img_size
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_src = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(mask_src, str):
            mask = cv2.imread(mask_src, 0)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = mask_src.copy()
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype("float32")
        mask = np.expand_dims(mask, axis=0)
        return img, mask, img_path

# ----------------------------- TRANSFORMS -----------------------------
train_transforms = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.2),
    ], p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.2),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

valid_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ----------------------------- BUILD DATASET + SPLIT -----------------------------
image_dirs = [TRAIN1_IMAGES, TRAIN2_IMAGES]
xml_paths = []
if os.path.exists(TRAIN1_XML):
    xml_paths.append(TRAIN1_XML)
if os.path.exists(TRAIN2_XML):
    xml_paths.append(TRAIN2_XML)

full_dataset = FarmlandDataset(image_dirs=image_dirs, mask_folder=MASK_FOLDER, xml_paths=xml_paths, transforms=None, img_size=IMAGE_SIZE)

if len(full_dataset) < 1:
    raise RuntimeError("No training samples found. Check image/mask/XML paths.")

val_pct = 0.2
val_size = max(1, int(len(full_dataset) * val_pct))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

class WrappedDataset(Dataset):
    def __init__(self, base_ds, transforms):
        self.base = base_ds
        self.transforms = transforms
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, mask, img_path = self.base[idx]
        augmented = self.transforms(image=img, mask=mask[0]*255)
        img_t = augmented['image']
        mask_t = augmented['mask']
        mask_t = (mask_t > 127).astype("float32")
        mask_t = np.expand_dims(mask_t, axis=0)
        return img_t, mask_t, img_path

train_dataset = WrappedDataset(train_ds, train_transforms)
val_dataset = WrappedDataset(val_ds, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------------------- MODEL -----------------------------
print("Loading DeepLabV3+ (resnet101 backbone)")
model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(DEVICE)

# ----------------------------- LOSS + OPTIMIZER -----------------------------
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")

def combined_loss(outputs, targets, bce_weight=0.5):
    bce = bce_loss(outputs, targets)
    probs = torch.sigmoid(outputs)
    d_loss = dice_loss(probs, targets)
    return bce * bce_weight + d_loss * (1 - bce_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ----------------------------- METRICS -----------------------------
def calc_metrics_numpy(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = 1.0 if union == 0 else intersection / union
    total = y_true.sum() + y_pred.sum()
    dice = 1.0 if total == 0 else (2 * intersection) / total
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return iou, dice, precision, recall

# ----------------------------- TRAIN + VALIDATE -----------------------------
best_val_dice = -1
history = {'train_loss':[], 'val_loss':[], 'val_dice':[], 'val_iou':[]}

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train", leave=False)
    for imgs, masks, _ in pbar:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        masks = masks.to(DEVICE, dtype=torch.float)
        preds = model(imgs)
        loss = combined_loss(preds, masks, bce_weight=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())
    avg_train_loss = running_loss / len(train_loader.dataset)
    model.eval()
    val_loss = 0.0
    dices = []
    ious = []
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)
            preds = model(imgs)
            loss = combined_loss(preds, masks, bce_weight=0.5)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(preds).cpu().numpy()
            gt = masks.cpu().numpy()
            for i in range(probs.shape[0]):
                p = (probs[i,0] > 0.5).astype(np.uint8)
                g = (gt[i,0] > 0.5).astype(np.uint8)
                iou, dice, prec, rec = calc_metrics_numpy(g, p)
                dices.append(dice); ious.append(iou)
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_dice = np.mean(dices) if dices else 0.0
    avg_val_iou = np.mean(ious) if ious else 0.0
    scheduler.step(avg_val_loss)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)
    print(f"Epoch {epoch}/{EPOCHS} TrainLoss: {avg_train_loss:.4f} ValLoss: {avg_val_loss:.4f} ValDice: {avg_val_dice:.4f} ValIoU: {avg_val_iou:.4f}")
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': best_val_dice,
        }, MODEL_PATH)
        print(f"Saved best model (ValDice={best_val_dice:.4f}) to {MODEL_PATH}")

# ----------------------------- PLOT LOSS -----------------------------
plt.figure(figsize=(10,5))
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.title('Training / Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------- LOAD BEST MODEL -----------------------------
print("Loading best model for prediction...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----------------------------- PREDICTION -----------------------------
predict_images = sorted(glob(os.path.join(PREDICT_IMAGES, "*.jpg")) + glob(os.path.join(PREDICT_IMAGES, "*.png")) + glob(os.path.join(PREDICT_IMAGES, "*.jpeg")))
print("Total images to predict:", len(predict_images))

transform_predict = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

with torch.no_grad():
    for img_path in tqdm(predict_images, desc="Predicting"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig = img_bgr.copy()
        h,w = orig.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = transform_predict(image=img_rgb)
        tensor = aug['image'].unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_resized = cv2.resize(mask, (w,h))
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = base + "_pred.png"
        cv2.imwrite(os.path.join(PREDICTION_FOLDER, mask_name), mask_resized)
        overlay = orig.copy()
        overlay[mask_resized>0] = (0,0,255)
        cv2.imwrite(os.path.join(OVERLAY_FOLDER, mask_name), overlay)

print("Predictions saved to", PREDICTION_FOLDER, "and", OVERLAY_FOLDER)

# ----------------------------- METRICS EVALUATION -----------------------------
gt_paths = glob(os.path.join(MASK_FOLDER, "*.png")) + glob(os.path.join(MASK_FOLDER, "*.jpg"))
pred_paths = glob(os.path.join(PREDICTION_FOLDER, "*_pred.png"))
pred_dict = { normalize_name(os.path.basename(p)): p for p in pred_paths }

ious = []; dices = []; precisions = []; recalls = []
matched = 0
for gt_path in gt_paths:
    key = normalize_name(os.path.basename(gt_path))
    if key not in pred_dict:
        continue
    pred_path = pred_dict[key]
    matched += 1
    gt = cv2.imread(gt_path, 0)
    pred = cv2.imread(pred_path, 0)
    if gt is None or pred is None:
        continue
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    iou, dice, prec, rec = calc_metrics_numpy(gt_bin, pred_bin)
    ious.append(iou); dices.append(dice); precisions.append(prec); recalls.append(rec)

print("Matched GT/pred samples:", matched)
avg_iou = np.mean(ious) if ious else 0.0
avg_dice = np.mean(dices) if dices else 0.0
avg_prec = np.mean(precisions) if precisions else 0.0
avg_rec = np.mean(recalls) if recalls else 0.0

print(f"IoU: {avg_iou:.4f}")
print(f"Dice: {avg_dice:.4f}")
print(f"Precision: {avg_prec:.4f}")
print(f"Recall: {avg_rec:.4f}")

# ----------------------------- METRICS BAR PLOT -----------------------------
metrics = ["IoU","Dice","Precision","Recall"]
values = [avg_iou, avg_dice, avg_prec, avg_rec]
plt.figure(figsize=(7,4))
bars = plt.bar(metrics, values, color=["#4C72B0","#55A868","#C44E52","#8172B3"])
plt.ylim(0,1)
for bar,v in zip(bars,values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha='center')
plt.title("Segmentation Metrics")
plt.show()

print("Pipeline complete.")
