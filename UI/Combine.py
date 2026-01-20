import numpy as np
import cv2

# 可选：更高效的向量化版本
def combine_masks(mask_show, std_img, alpha=0.5, brightness_factor=1.0, mask_threshold=10):
    """
    Overlay masks on the target video
    """

    assert mask_show.shape[:2] == std_img.shape, "unmached input image shape"
    assert 0 <= alpha <= 1, "alpha between 0 and 1"
    assert brightness_factor > 0, "Brightness factor > 0"
    

    std_img_bright = np.clip(std_img * brightness_factor, 0, 255)
    if std_img_bright.dtype != np.uint8:
        std_img_bright = std_img_bright.astype(np.uint8)
    

    gray_rgb = cv2.cvtColor(std_img_bright, cv2.COLOR_GRAY2RGB)
    
 
    if mask_show.dtype != np.uint8:
        mask_show = mask_show.astype(np.uint8)
    
    mask_gray = cv2.cvtColor(mask_show, cv2.COLOR_RGB2GRAY)
    mask_area = mask_gray > mask_threshold
    

    mask_alpha_3d = np.zeros_like(mask_show, dtype=np.float32)
    mask_alpha_3d[mask_area] = alpha
    

    result = (1 - mask_alpha_3d) * gray_rgb + mask_alpha_3d * mask_show
    

    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result