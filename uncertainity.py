import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import ISICDataset
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ISICDataset("data/val_input", "data/val_maskss")

model = UNet().to(device)
model.load_state_dict(torch.load("outputs/model_epoch_35.pth"))

# Enable dropout during inference
model.train()

img, mask = dataset[0]
img = img.unsqueeze(0).to(device)

T = 15
preds = []

with torch.no_grad():
    for _ in range(T):
        pred = model(img)
        preds.append(pred.cpu().numpy())

preds = np.array(preds).squeeze()

mean_pred = preds.mean(axis=0)
variance = preds.var(axis=0)

# Threshold
mean_bin = (mean_pred > 0.5).astype(int)

# Plot
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(img.squeeze().permute(1,2,0).cpu())
plt.title("Image")

plt.subplot(1,4,2)
plt.imshow(mean_bin, cmap='gray')
plt.title("Prediction")

plt.subplot(1,4,3)
plt.imshow(variance, cmap='hot')
plt.title("Uncertainty")

plt.subplot(1,4,4)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title("Ground Truth")

plt.show()