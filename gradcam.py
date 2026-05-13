import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.dataset import ISICDataset
from models.unet import UNet

# =========================
# CUSTOM TARGET FOR SEGMENTATION
# =========================
class SegmentationTarget:
    def __call__(self, model_output):
        return model_output.mean()

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATASET
# =========================
dataset = ISICDataset(
    r"D:\data\val_input",
    r"D:\data\val_masks"
)

# =========================
# LOAD MODEL
# =========================
model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "outputs/model_epoch_35.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()

# =========================
# LOAD SAMPLE
# =========================
img, mask = dataset[0]

input_tensor = img.unsqueeze(0).to(device)

# =========================
# TARGET LAYER
# =========================
# Deepest semantic layer
target_layers = [model.bottleneck]

# =========================
# CREATE GRADCAM OBJECT
# =========================
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# =========================
# GENERATE GRADCAM
# =========================
targets = [SegmentationTarget()]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

# =========================
# PREPARE IMAGE
# =========================
rgb_img = img.permute(1, 2, 0).cpu().numpy()

# Normalize image
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# =========================
# OVERLAY HEATMAP
# =========================
visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# =========================
# MODEL PREDICTION
# =========================
with torch.no_grad():

    pred = model(input_tensor)

    pred = pred.squeeze().cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(20,5))

# ORIGINAL IMAGE
plt.subplot(1,4,1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis("off")

# PREDICTED MASK
plt.subplot(1,4,2)
plt.imshow(pred_bin, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

# GRADCAM HEATMAP
plt.subplot(1,4,3)
plt.imshow(grayscale_cam, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis("off")

# OVERLAY
plt.subplot(1,4,4)
plt.imshow(visualization)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()