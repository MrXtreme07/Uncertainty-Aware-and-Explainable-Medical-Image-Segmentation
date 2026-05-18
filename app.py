import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

from inference import segment_uploaded_image
from uncertainity import compute_uncertainty
from gradcam import generate_gradcam
from explainability import generate_explanation

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Medical AI Dashboard",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("Uncertainty-Aware Medical Image Segmentation")

st.markdown("""
This dashboard performs:
- Medical Image Segmentation
- MC Dropout Uncertainty Estimation
- GradCAM Explainability
- Rule-Based Explainability
""")

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

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
    # LOAD IMAGE
    # =================================================

    image = Image.open(uploaded_file).convert("RGB")

    input_image = transform(image)

    input_tensor = input_image.unsqueeze(0)

    # =================================================
    # RGB IMAGE
    # =================================================

    rgb_img = input_image.permute(1,2,0).numpy()

    rgb_img = (
        rgb_img - rgb_img.min()
    ) / (
        rgb_img.max() - rgb_img.min()
    )

    # =================================================
    # SEGMENTATION
    # =================================================

    pred, pred_bin = segment_uploaded_image(
        input_tensor,
        threshold
    )

    # =================================================
    # UNCERTAINTY
    # =================================================

    mean_pred, variance, reliability_percent = (
        compute_uncertainty(
            input_tensor,
            T
        )
    )

    # =================================================
    # GRADCAM
    # =================================================

    grayscale_cam, gradcam_overlay = (
        generate_gradcam(
            input_tensor,
            rgb_img
        )
    )

    # =================================================
    # RULE-BASED EXPLAINABILITY
    # =================================================

    explanation = generate_explanation(
        variance,
        reliability_percent
    )

    # =================================================
    # UNCERTAINTY OVERLAY
    # =================================================

    fig_overlay, ax_overlay = plt.subplots(figsize=(5,5))

    ax_overlay.imshow(rgb_img)

    ax_overlay.imshow(
        variance,
        cmap='inferno',
        alpha=0.5
    )

    ax_overlay.set_title("Uncertainty Overlay")

    ax_overlay.axis("off")

    # =================================================
    # DASHBOARD TITLE
    # =================================================

    st.subheader(" Clinical Dashboard")

    # =================================================
    # FIRST ROW
    # =================================================

    col1, col2, col3 = st.columns(3)

    # ORIGINAL IMAGE
    with col1:

        st.image(
            rgb_img,
            caption="Original Image",
            use_container_width=True
        )

    # PREDICTED MASK
    with col2:

        fig_mask, ax_mask = plt.subplots(figsize=(5,5))

        ax_mask.imshow(pred_bin, cmap='gray')

        ax_mask.set_title("Predicted Mask")

        ax_mask.axis("off")

        st.pyplot(fig_mask)

    # RAW PREDICTION
    with col3:

        fig_pred, ax_pred = plt.subplots(figsize=(5,5))

        ax_pred.imshow(mean_pred, cmap='gray')

        ax_pred.set_title("Raw Prediction")

        ax_pred.axis("off")

        st.pyplot(fig_pred)

    # =================================================
    # SECOND ROW
    # =================================================

    col4, col5, col6 = st.columns(3)

    # UNCERTAINTY MAP
    with col4:

        fig_var, ax_var = plt.subplots(figsize=(5,5))

        ax_var.imshow(variance, cmap='inferno')

        ax_var.set_title("Uncertainty Map")

        ax_var.axis("off")

        st.pyplot(fig_var)

    # UNCERTAINTY OVERLAY
    with col5:

        st.pyplot(fig_overlay)

    # GRADCAM HEATMAP
    with col6:

        fig_cam, ax_cam = plt.subplots(figsize=(5,5))

        ax_cam.imshow(grayscale_cam, cmap='jet')

        ax_cam.set_title("GradCAM Heatmap")

        ax_cam.axis("off")

        st.pyplot(fig_cam)

    # =================================================
    # THIRD ROW
    # =================================================

    col7, col8 = st.columns(2)

    # GRADCAM OVERLAY
    with col7:

        st.image(
            gradcam_overlay,
            caption="GradCAM Overlay",
            use_container_width=True
        )

    # METRICS
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

        # =============================================
        # DETAILED RELIABILITY STATUS
        # =============================================

        if reliability_percent < 70:

            st.error(
                "VERY LOW RELIABILITY"
            )

        elif reliability_percent < 80:

            st.warning(
                "LOW RELIABILITY"
            )

        elif reliability_percent < 85:

            st.warning(
                "MODERATE RELIABILITY"
            )

        elif reliability_percent < 90:

            st.info(
                "GOOD RELIABILITY"
            )

        elif reliability_percent < 95:

            st.success(
                "VERY HIGH RELIABILITY"
            )

        else:

            st.success(
                "EXCELLENT RELIABILITY"
            )

    # =================================================
    # RULE-BASED EXPLAINABILITY PANEL
    # =================================================

    st.subheader(" Explainability")

    st.info(explanation)