import torch
import numpy as np

from inference import model, device

# =====================================================
# MC DROPOUT UNCERTAINTY
# =====================================================

def compute_uncertainty(input_tensor, T=15):

    input_tensor = input_tensor.to(device)

    model.train()

    preds = []

    with torch.no_grad():

        for _ in range(T):

            pred = model(input_tensor)

            preds.append(pred.cpu().numpy())

    preds = np.array(preds).squeeze()

    mean_pred = preds.mean(axis=0)

    variance = preds.var(axis=0)

    mean_uncertainty = variance.mean()

    normalized_uncertainty = min(
        mean_uncertainty * 50,
        1
    )

    reliability = 1 - normalized_uncertainty

    reliability_percent = reliability * 100
    return (
        mean_pred,
        variance,
        reliability_percent
    )