# Implementation Plan: Uncertainty-Aware and Explainable Medical Image Segmentation

## Objective
Build a medical image segmentation pipeline that produces:
- Segmentation mask
- Uncertainty map (via MC Dropout)
- Explanation map (via Grad-CAM)

and analyze whether uncertainty correlates with model errors.

---

## Project Structure

```
project/
├── data/
├── models/
├── train.py
├── inference.py
├── utils/
├── outputs/
```

---

## Module 0 — Environment Setup

### Tasks
- Install dependencies: PyTorch, torchvision, numpy, matplotlib
- Configure GPU if available

### Checks
- `torch.cuda.is_available()` returns True (if GPU expected)
- No import errors

### Pitfalls
- CUDA/version mismatch
- Incorrect environment setup

---

## Module 1 — Data Pipeline

### Tasks
- Load ISIC images and masks
- Resize (e.g., 256×256)
- Normalize images
- Convert masks to binary (0/1)

### Output
- Image: (3, H, W)
- Mask: (1, H, W)

### Checks
- Visualize image-mask overlay
- Mask values are {0,1}
- Correct pairing of images and masks

### Pitfalls
- Masks in {0,255} instead of {0,1}
- Misaligned image-mask pairs
- Distorted resizing

---

## Module 2 — U-Net Model

### Tasks
- Implement or import U-Net
- Include dropout layers

### Output
- Model output: (1, H, W)

### Checks
- Forward pass works
- Output size matches input
- No NaNs in output

### Pitfalls
- Shape mismatch in skip connections
- Missing sigmoid activation

---

## Module 3 — Training

### Tasks
- Loss: BCE or Dice
- Optimizer: Adam
- Train for multiple epochs

### Outputs
- Trained model
- Loss curve

### Checks
- Loss decreases over time
- Predictions improve visually

### Pitfalls
- Loss not decreasing (bug or bad config)
- Incorrect mask dtype
- Learning rate issues

---

## Module 4 — Baseline Evaluation

### Tasks
- Predict on validation set
- Compute Dice score

Dice = 2|P ∩ G| / (|P| + |G|)

### Outputs
- Dice score
- Sample predictions

### Checks
- Dice score > ~0.6
- Predictions resemble lesions

### Pitfalls
- Using logits instead of probabilities
- Incorrect thresholding

---

## Module 5 — Uncertainty (MC Dropout)

### Tasks
- Enable dropout during inference
- Run T forward passes (T = 10–20)
- Compute:
  - Mean prediction
  - Variance map

### Outputs
- Mean map
- Variance (uncertainty) map

### Checks
- Variance not zero everywhere
- Outputs differ across passes
- Higher variance near boundaries

### Pitfalls
- Dropout not active at inference
- Too few samples
- Identical predictions

---

## Module 6 — Error Map

### Tasks
- Compute pixel-wise error:
  error = |prediction - ground_truth|

### Outputs
- Error map

### Checks
- Errors concentrated near boundaries
- Matches visible segmentation mistakes

### Pitfalls
- Using probabilities instead of binary masks
- Misalignment issues

---

## Module 7 — Explainability (Grad-CAM)

### Tasks
- Apply Grad-CAM on final convolutional layer
- Generate heatmaps

### Outputs
- Explanation map

### Checks
- Highlights lesion regions
- Not random or empty

### Pitfalls
- Wrong layer selection
- Weak gradients

---

## Module 8 — Core Analysis

### Tasks
- Compare uncertainty map and error map
- Compute correlation (optional)

### Outputs
- Visual overlays
- Correlation score

### Checks
- High uncertainty aligns with high error regions
- Strong patterns near boundaries

### Pitfalls
- Weak or no correlation
- Poor visualization

---

## Module 9 — Visualization

### For each sample:
- Input image
- Ground truth mask
- Predicted mask
- Uncertainty map
- Error map
- Grad-CAM heatmap

### Checks
- Clear, interpretable outputs
- Proper scaling and overlays

---

## Master Checklist

- [ ] Data correctly loaded and aligned
- [ ] Model trains and loss decreases
- [ ] Dice score reasonable
- [ ] Uncertainty map meaningful
- [ ] Error map correct
- [ ] Correlation between uncertainty and error observed
- [ ] Visual outputs interpretable

---

## Execution Strategy

- Implement one module at a time
- Validate before moving forward
- Save intermediate outputs
- Avoid parallel debugging across modules

---

## Success Criteria

- Model produces reasonable segmentation
- Uncertainty highlights uncertain regions
- Uncertainty correlates with segmentation errors
- Visual evidence supports conclusions

---

## Key Risks

- Poor segmentation performance
- Dropout not functioning correctly
- No observable correlation
- Overcomplicating architecture or dataset

---

## Final Goal

Demonstrate that:

"The model’s uncertainty estimates provide meaningful signals about where it is likely to make errors."