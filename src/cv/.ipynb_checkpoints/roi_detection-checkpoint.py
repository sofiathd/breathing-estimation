from PIL import Image
import cv2
import torchvision.transforms as tvT
import numpy as np
import torch
from src.models import get_pifpaf_predictor, get_deeplab_model, device

_DEEPLAB_TRANSFORM = tvT.Compose([
    tvT.ToTensor(),
    tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def hardcoded_roi(camera, region):
    """
    Returns a hardcoded region of interest, depending on the camera.

    Args:
        camera (str): Camera identifier ("gray", "olympus", "flir", or "phone").
        region (str): Target ROI region ("chest" or "abdomen").

    Returns:
        tuple:
            - int: ROI top-left x-coordinate (pixels).
            - int: ROI top-left y-coordinate (pixels).
            - int: ROI width (pixels).
            - int: ROI height (pixels).
    """
    if camera == "gray":
        return (1200, 2000, 1100, 800) if region == "abdomen" else (1200, 1200, 1100, 800)
    elif camera == "olympus":
        return (750, 775, 500, 275) if region == "abdomen" else (750, 500, 500, 275)
    elif camera == "flir":
        return (200, 375, 225, 125) if region == "abdomen" else (200, 250, 225, 125)
    else:
        return (350, 1450, 300, 225) if region == "abdomen" else (350, 1175, 300, 225)

def detect_ROI(image_path, camera, region="chest", image_type='RGB', conf_thr=0.1):
    """
    Returns a torso region of interest in an image, using OpenPifPaf keypoints.
    The function loads an image, runs OpenPifPaf pose estimation, and builds an ROI around
    the shoulder keypoints. If detection fails or the shoulder confidence is too low, a
    camera-specific hardcoded ROI is returned as a fallback.

    Args:
        image_path (str): Path to the input image file.
        camera (str): Camera identifier used for hardcoded fallback ROIs.
        region (str): Target ROI region ("chest" or "abdomen").
        image_type (str): PIL conversion mode.
        conf_thr (float): Minimum confidence threshold for shoulder keypoints.

    Returns:
        tuple:
            - int: ROI top-left x-coordinate (pixels).
            - int: ROI top-left y-coordinate (pixels).
            - int: ROI width (pixels).
            - int: ROI height (pixels).
    """
    pil_im = Image.open(image_path).convert(image_type)
    predictor = get_pifpaf_predictor()
    predictions, _, _ = predictor.pil_image(pil_im)
    W, H = pil_im.size

    if predictions is None or len(predictions) == 0:
        return hardcoded_roi(camera, region)

    person = predictions[0]
    key_points = person.data
    ls, rs = key_points[5], key_points[6]

    if ls[2] < conf_thr or rs[2] < conf_thr:
        return hardcoded_roi(camera, region)
                
    else:
        shoulder_y = (ls[1] + rs[1]) / 2.0
        x_left, x_right = float(min(ls[0], rs[0])), float(max(ls[0], rs[0]))
        width = max(1.0, x_right - x_left)
        left, right = x_left, x_right 
        top, bottom = shoulder_y, shoulder_y + 1.2 * width

    left, right = int(max(0, min(left, W-1))), int(max(left+1, min(right, W)))
    top, bottom = int(max(0, min(top, H-1))), int(max(top+1, min(bottom, H)))
    
    h_box = bottom - top
    half_h = int(max(1, h_box // 2))
    w_box = right - left

    if region == "chest": return (left, top, w_box, half_h)
    elif region == "abdomen": return (left, top + half_h, w_box, half_h)
    return (left, top, w_box, half_h)

def segment_person_deeplab(frame_bgr):
    """
    Segment the person from an image using a DeepLab semantic segmentation model.
    The frame is converted to RGB, normalized with ImageNet statistics, and passed through
    DeepLab. A binary mask is produced by selecting the "person" class and post-processed
    with morphological closing to fill small holes.

    Args:
        frame_bgr (np.ndarray): Input frame in BGR format.

    Returns:
        mask (np.ndarray): Binary segmentation mask with values in {0, 255}, 
                        where 255 corresponds to the person region.
    """
    model = get_deeplab_model()
    
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = _DEEPLAB_TRANSFORM(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad(): 
        out = model(inp)["out"]
        
    labels = out.argmax(1).squeeze().cpu().numpy()
    mask = (labels == 15).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask
