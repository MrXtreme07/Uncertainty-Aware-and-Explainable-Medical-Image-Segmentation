import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(f for f in os.listdir(image_dir) if f.endswith((".jpg",".png",".jpeg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Image Path
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Safety check
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Load Image and Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image / 255.0                       # Normalize

        mask = (mask > 127).astype(np.float32)      # Binary Mask

        image = cv2.resize(image, (256, 256))       # Resize Image
        mask = cv2.resize(mask, (256, 256))

        image = np.transpose(image, (2,0,1))        # (3,H,W)
        mask = np.expand_dims(mask, axis=0)         # (1,H,W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
