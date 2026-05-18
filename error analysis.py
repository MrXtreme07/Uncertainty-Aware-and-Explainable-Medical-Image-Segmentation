import numpy as np

# =====================================================
# ERROR MAP
# =====================================================

def compute_error_map(pred_bin, gt_mask):

    pred_bin = pred_bin / 255

    error_map = np.abs(
        pred_bin.astype(np.float32)
        -
        gt_mask.astype(np.float32)
    )

    return error_map