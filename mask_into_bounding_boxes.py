import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIRECTORY = BASE_DIR / "img"


plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# --- Cargar imagen (PIL → tensor)
img_path = ASSETS_DIRECTORY / "FudanPed00054.png"
img = Image.open(img_path).convert("RGB")
img = F.to_tensor(img)  # float [0,1]
img_uint8 = (img * 255).to(torch.uint8)

# --- Cargar máscara
mask_path = ASSETS_DIRECTORY / "FudanPed00054_mask.png"
mask = Image.open(mask_path)
mask = torch.as_tensor(np.array(mask), dtype=torch.int64)


print(mask.size())
print(img.size())
print(mask)


# --- Obtener IDs de objetos
obj_ids = torch.unique(mask)
obj_ids = obj_ids[1:]  # quitar fondo

# --- Crear máscaras booleanas
masks = mask == obj_ids[:, None, None]


print(masks.size())
print(masks)


# --- Dibujar máscaras
drawn_masks = [
    draw_segmentation_masks(img_uint8, m, alpha=0.8, colors="blue") for m in masks
]
show(drawn_masks)


# --- Bounding boxes
boxes = masks_to_boxes(masks)
drawn_boxes = draw_bounding_boxes(img_uint8, boxes, colors="red")
show(drawn_boxes)

# --- Modelo (INFERENCIA)
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
print(img.size())

transforms = weights.transforms()
img_model = transforms(img)

model.eval()
with torch.no_grad():
    outputs = model([img_model])

print(outputs)
