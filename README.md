# Uncertainty-Aware and Explainable Medical Image Segmentation

## Overview

This project presents an uncertainty-aware and explainable deep learning system for skin lesion segmentation using U-Net, MC Dropout, and GradCAM. The system performs lesion segmentation while also estimating prediction uncertainty and visualizing model attention regions for improved interpretability and trust in medical AI.

---

# Features

- U-Net based skin lesion segmentation
- MC Dropout uncertainty estimation
- GradCAM explainability heatmaps
- Reliability score generation
- Interactive Streamlit dashboard
- Real-time visualization and analysis

---

# Pipeline

Input Image  
→ Preprocessing  
→ U-Net Encoder  
→ U-Net Decoder  
→ MC Dropout Inference  
→ Uncertainty Map  
→ GradCAM Heatmap  
→ Reliability Analysis  
→ Dashboard Outputs

---

# Dataset

The project uses the **ISIC (International Skin Imaging Collaboration)** dermoscopy dataset containing:
- RGB skin lesion images
- Binary segmentation masks

---

# Preprocessing

- Resize images to `256×256`
- Convert to tensors
- Normalize pixel values to `[0,1]`
- Add batch dimension

---

# Model Architecture

## U-Net
- Encoder-decoder segmentation architecture
- Skip connections for preserving spatial details
- Double convolution blocks with ReLU and Dropout

## MC Dropout
Dropout remains active during inference to generate multiple stochastic predictions for uncertainty estimation.

## GradCAM
Generates heatmaps showing image regions influencing segmentation predictions.

---

# Key Mathematics

## Convolution

(I * K)[i,j] = Σm Σn I[i+m, j+n] · K[m,n]

---

## Binary Cross Entropy Loss

BCE = -[y·log(p̂) + (1-y)·log(1-p̂)]

---

## MC Dropout Mean & Variance

μ = (1/T) Σ f_t(x)

σ² = (1/T) Σ (f_t - μ)²

---

## GradCAM

Lᶜ = ReLU(Σ αₖ Aᵏ)

---

# Outputs

- Segmentation Mask
- Uncertainty Map
- GradCAM Overlay
- Reliability Report

---

# Project Structure

```text
├── app.py
├── train.py
├── inference.py
├── uncertainty.py
├── gradcam.py
├── explainability.py
├── requirements.txt
├── README.md
├── models/
├── outputs/
├── utils/