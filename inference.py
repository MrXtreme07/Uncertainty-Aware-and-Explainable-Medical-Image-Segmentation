import torch
import matplotlib.pyplot as plt
from utils.dataset import ISICDataset
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded Dataset for Validation
dataset = ISICDataset("data/val_input", "data/val_masks")

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("outputs/model_epoch_35.pth"))
model.eval()

# Get one sample
img, mask = dataset[0]
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img)

pred = pred.squeeze().cpu().numpy()
mask = mask.squeeze().numpy()

#Threshold
pred_bin = (pred > 0.5).astype(int)

# Plot
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(img.squeeze().permute(1,2,0).cpu())
plt.title("Image")

plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred_bin, cmap='gray')
plt.title("Prediction")

plt.show()

# Calculate Dice Score
def dice_score(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

dice = dice_score(pred_bin, mask)
print("Dice Score: ", dice)