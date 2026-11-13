from .rife_image_interpolation import RIFEFrameInterpolate
from .rife_stream_interpolation import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS["RIFEFrameInterpolate"] = RIFEFrameInterpolate
NODE_DISPLAY_NAME_MAPPINGS["RIFEFrameInterpolate"] = "RIFE Frame Interpolation (IMAGE In/Out)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
