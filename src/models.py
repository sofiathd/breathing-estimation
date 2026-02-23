import openpifpaf
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch

_PIFPAF_PREDICTOR = None
_DEEPLAB_MODEL = None
_COTRACKER = None

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pifpaf_predictor():
    global _PIFPAF_PREDICTOR
    if _PIFPAF_PREDICTOR is None:
        _PIFPAF_PREDICTOR = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    return _PIFPAF_PREDICTOR

def get_deeplab_model():
    global _DEEPLAB_MODEL
    if _DEEPLAB_MODEL is None:
        _DEEPLAB_MODEL = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
        _DEEPLAB_MODEL.eval()
    return _DEEPLAB_MODEL

def get_cotracker():
    global _COTRACKER
    if _COTRACKER is None:
        _COTRACKER = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", map_location=device).to(device)
        _COTRACKER.eval()
    return _COTRACKER
