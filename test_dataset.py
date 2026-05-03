from utils.dataset import ISICDataset
import matplotlib.pyplot as plt

dataset = ISICDataset(
    "data/training_input",
    "data/train_masks"
)

img, mask = dataset[0]

print("Image Shape: ", img.shape)
print("Mask Shape: ", mask.shape)

plt.subplot(1,2,1)
plt.imshow(img.permute(1,2,0))
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title("Mask")

plt.show()