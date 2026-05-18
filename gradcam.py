import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from inference import model, device

# =====================================================
# CUSTOM TARGET
# =====================================================

class SegmentationTarget:

    def __call__(self, model_output):

        return model_output.mean()

# =====================================================
# GENERATE GRADCAM
# =====================================================

def generate_gradcam(input_tensor, rgb_img):

    input_tensor = input_tensor.to(device)

    model.eval()

    target_layers = [model.bottleneck]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [SegmentationTarget()]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return grayscale_cam, visualization