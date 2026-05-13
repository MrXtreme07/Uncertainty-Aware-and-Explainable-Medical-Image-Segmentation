import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.unet import UNet

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Medical AI Dashboard",
    layout="wide"
)

st.title("🧠 Uncertainty-Aware Medical Image Segmentation")

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            "outputs/model_epoch_35.pth",
            map_location=torch.device('cpu')
        )
    )

    return model

model = load_model()

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# =====================================================
# SEGMENTATION TARGET FOR GRADCAM
# =====================================================
class SegmentationTarget:
    def __call__(self, model_output):
        return model_output.mean()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Controls")

threshold = st.sidebar.slider(
    "Segmentation Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

T = st.sidebar.slider(
    "MC Dropout Passes",
    5,
    30,
    15,
    1
)

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Dermoscopy Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# MAIN PIPELINE
# =====================================================
if uploaded_file:

    # =================================================
    # IMAGE LOADING
    # =================================================
    image = Image.open(uploaded_file).convert("RGB")

    input_image = transform(image)

    input_tensor = input_image.unsqueeze(0).to(device)

    # =================================================
    # SEGMENTATION
    # =================================================
    model.eval()

    with torch.no_grad():

        pred = model(input_tensor)

    pred = pred.squeeze().cpu().numpy()

    pred_bin = (pred > threshold).astype(np.uint8)

    # =================================================
    # MC DROPOUT UNCERTAINTY
    # =================================================
    model.train()

    preds = []

    with torch.no_grad():

        for _ in range(T):

            p = model(input_tensor)

            preds.append(p.cpu().numpy())

    preds = np.array(preds).squeeze()

    mean_pred = preds.mean(axis=0)

    variance = preds.var(axis=0)

    # =================================================
    # RELIABILITY SCORE
    # =================================================
    reliability = 1 - variance.mean()

    reliability_percent = reliability * 100

    # =================================================
    # GRADCAM
    # =================================================
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

    # =================================================
    # IMAGE PREP
    # =================================================
    rgb_img = input_image.permute(1, 2, 0).cpu().numpy()

    rgb_img = (
        rgb_img - rgb_img.min()
    ) / (
        rgb_img.max() - rgb_img.min()
    )

    # =================================================
    # GRADCAM OVERLAY
    # =================================================
    gradcam_overlay = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # =================================================
    # UNCERTAINTY OVERLAY
    # =================================================
    fig_uncertainty, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(rgb_img)

    ax.imshow(
        variance,
        cmap='inferno',
        alpha=0.5
    )

    ax.set_title("Uncertainty Overlay")

    ax.axis("off")

    # =================================================
    # DASHBOARD
    # =================================================
    st.subheader("📊 Clinical AI Dashboard")

    # =================================================
    # FIRST ROW
    # =================================================
    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            rgb_img,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            pred_bin,
            caption="Predicted Mask",
            use_container_width=True,
            clamp=True
        )

    with col3:

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))

        ax_pred.imshow(mean_pred, cmap='gray')

        ax_pred.set_title("Raw Prediction")

        ax_pred.axis("off")

        st.pyplot(fig_pred)

    # =================================================
    # SECOND ROW
    # =================================================
    col4, col5, col6 = st.columns(3)

    with col4:

        fig_var, ax_var = plt.subplots(figsize=(5, 5))

        ax_var.imshow(variance, cmap='inferno')

        ax_var.set_title("Uncertainty Map")

        ax_var.axis("off")

        st.pyplot(fig_var)

    with col5:

        st.pyplot(fig_uncertainty)

    with col6:

        fig_cam, ax_cam = plt.subplots(figsize=(5, 5))

        ax_cam.imshow(grayscale_cam, cmap='jet')

        ax_cam.set_title("Grad-CAM Heatmap")

        ax_cam.axis("off")

        st.pyplot(fig_cam)

    # =================================================
    # THIRD ROW
    # =================================================
    col7, col8 = st.columns(2)

    with col7:

        st.image(
            gradcam_overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    with col8:

        st.subheader("📈 Metrics")

        st.metric(
            "Reliability Score",
            f"{reliability_percent:.2f}%"
        )

        st.metric(
            "Mean Uncertainty",
            f"{variance.mean():.6f}"
        )

        st.metric(
            "Max Uncertainty",
            f"{variance.max():.6f}"
        )

        if reliability_percent > 85:
            st.success("HIGH CONFIDENCE PREDICTION")

        elif reliability_percent > 70:
            st.warning("MODERATE CONFIDENCE")

        else:
            st.error("LOW CONFIDENCE PREDICTION")