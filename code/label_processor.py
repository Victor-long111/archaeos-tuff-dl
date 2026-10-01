# label_processor.py
import numpy as np

IGNORE_VAL = 255

def mask_to_target_and_valid(mask: np.ndarray, ignore_val: int = IGNORE_VAL, pos_val: int = 1):
    """
    Input:
      mask: (H,W) int array where:
            1 = positive (AT)
            0 = background
            255 = ignore
    Output:
      target: (1,H,W) float32 in {0,1}; ignored regions are set to 0.
      valid : (1,H,W) float32 in {0,1}; ignored regions are 0 and valid regions are 1.
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = mask.astype(np.int16)

    valid = (mask != ignore_val).astype(np.float32)
    target = ((mask == pos_val) & (mask != ignore_val)).astype(np.float32)

    # add channel dim to match logits shape (B,1,H,W)
    target = target[None, ...]
    valid = valid[None, ...]
    return target, valid

import scipy.ndimage as ndi

def mask_to_boundary(mask: np.ndarray, ignore_val: int = IGNORE_VAL, pos_val: int = 1, radius: int = 1):
    """Generate a thin boundary map (1 on boundary pixels, 0 elsewhere)."""
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = mask.astype(np.int16)

    valid = (mask != ignore_val)
    pos = (mask == pos_val) & valid

    # structuring element
    if radius <= 1:
        struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    else:
        struct = ndi.generate_binary_structure(2, 1)
        struct = ndi.iterate_structure(struct, radius)

    dil = ndi.binary_dilation(pos, structure=struct)
    ero = ndi.binary_erosion(pos, structure=struct, border_value=0)

    boundary = (dil ^ ero) & valid
    boundary = boundary.astype(np.float32)[None, ...]  # (1,H,W)
    return boundary

def mask_to_sdf(mask: np.ndarray, ignore_val: int = IGNORE_VAL, pos_val: int = 1, clip: float = 20.0):
    """
    Signed distance field to the object boundary, normalized to [-1,1] by `clip`.
    Positive inside the object, negative outside.
    Ignore pixels are set to 0 and excluded by valid mask.

    Returns:
      sdf: (1,H,W) float32 in [-1,1]
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = mask.astype(np.int16)

    valid = (mask != ignore_val)
    pos = (mask == pos_val) & valid

    # Boundary pixels
    boundary = mask_to_boundary(mask, ignore_val=ignore_val, pos_val=pos_val, radius=1)[0].astype(bool)

    # Guard: if no positive pixels or no boundary found, return zero SDF to avoid degenerate constant targets
    if pos.sum() == 0 or boundary.sum() == 0:
        sdf = np.zeros_like(mask, dtype=np.float32)
        sdf = np.where(valid, sdf, 0.0).astype(np.float32)[None, ...]
        return sdf

    # Distance to boundary (0 at boundary)
    dist = ndi.distance_transform_edt(~boundary).astype(np.float32)

    # sign: + inside, - outside
    sign = np.where(pos, 1.0, -1.0).astype(np.float32)
    sdf = dist * sign

    if clip is not None and clip > 0:
        sdf = np.clip(sdf / float(clip), -1.0, 1.0)

    # Ignore pixels: set sdf=0 but will be masked out by valid
    sdf = np.where(valid, sdf, 0.0).astype(np.float32)[None, ...]
    return sdf
