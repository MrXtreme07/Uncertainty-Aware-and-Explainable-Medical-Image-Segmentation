import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils.dataset import ISICDataset
from models.unet import UNet

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
dataset = ISICDataset(
    r"D:\data\val_input",
    r"D:\data\val_masks"
)
git
# =========================
# MODEL
# =========================
model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "outputs/model_epoch_35.pth",
        map_location=torch.device('cpu')
    )
)

# IMPORTANT:
# Keep dropout active for MC Dropout
model.train()

# =========================
# LOAD SAMPLE
# =========================
img, mask = dataset[0]

img_input = img.unsqueeze(0).to(device)

# Ground truth mask
gt_mask = mask.squeeze().numpy()

# =========================
# MC DROPOUT
# =========================
T = 15
preds = []

with torch.no_grad():

    for _ in range(T):

        # FORWARD PASS
        pred = model(img_input)

        # DO NOT APPLY SIGMOID AGAIN
        # Model already outputs probabilities

        preds.append(pred.cpu().numpy())

preds = np.array(preds).squeeze()

# =========================
# MEAN + VARIANCE
# =========================
mean_pred = preds.mean(axis=0)
variance = preds.var(axis=0)

# =========================
# DEBUGGING VALUES
# =========================
print("Min prediction :", mean_pred.min())
print("Max prediction :", mean_pred.max())
print("Mean prediction:", mean_pred.mean())

# =========================
# BINARY PREDICTION
# =========================
mean_bin = (mean_pred > 0.5).astype(np.uint8)

# =========================
# ERROR MAP
# =========================
error_map = np.abs(mean_bin - gt_mask)

# =========================
# CORRELATION ANALYSIS
# =========================
u = variance.flatten()
e = error_map.flatten()

corr, _ = pearsonr(u, e)

print(f"\nPearson Correlation between Uncertainty and Error: {corr:.4f}")

# =========================
# MEAN UNCERTAINTY ANALYSIS
# =========================
variance_flat = variance.flatten()

correct_uncertainty = variance_flat[e == 0].mean()
wrong_uncertainty = variance_flat[e == 1].mean()

print(f"Mean uncertainty in CORRECT regions : {correct_uncertainty:.6f}")
print(f"Mean uncertainty in WRONG regions   : {wrong_uncertainty:.6f}")

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(22,5))

# ORIGINAL IMAGE
plt.subplot(1,6,1)
plt.imshow(img.permute(1,2,0).cpu())
plt.title("Original Image")
plt.axis("off")

# RAW PREDICTION
plt.subplot(1,6,2)
plt.imshow(mean_pred, cmap='gray')
plt.title("Raw Prediction")
plt.axis("off")

# PREDICTED MASK
plt.subplot(1,6,3)
plt.imshow(mean_bin, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

# UNCERTAINTY MAP
plt.subplot(1,6,4)
plt.imshow(variance, cmap='hot')
plt.title("Uncertainty Map")
plt.axis("off")

# ERROR MAP
plt.subplot(1,6,5)
plt.imshow(error_map, cmap='hot')
plt.title("Error Map")
plt.axis("off")

# GROUND TRUTH
plt.subplot(1,6,6)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()