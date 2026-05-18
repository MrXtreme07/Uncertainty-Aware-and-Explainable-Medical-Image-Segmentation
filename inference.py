import torch
import numpy as np

from models.unet import UNet

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODEL
# =====================================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "outputs/model_epoch_35.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()

# =====================================================
# SEGMENTATION FUNCTION
# =====================================================

def segment_uploaded_image(input_tensor, threshold=0.5):

    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.no_grad():

        pred = model(input_tensor)

    pred = pred.squeeze().cpu().numpy()

    pred_bin = (
        (pred > threshold).astype(np.uint8)
    ) * 255

    return pred, pred_bin