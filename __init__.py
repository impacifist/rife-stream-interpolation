from .rife_image_interpolation import RIFEFrameInterpolate
from .rife_model_loader import RIFEModelLoader

NODE_CLASS_MAPPINGS = {
    "RIFEModelLoader": RIFEModelLoader,
    "RIFEFrameInterpolate": RIFEFrameInterpolate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFEModelLoader": "Load RIFE v4.26+ (BF16 / TF32 / FP32)",
    "RIFEFrameInterpolate": "RIFE Frame Interpolation",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
