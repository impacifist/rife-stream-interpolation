"""ComfyUI model loader for RIFE v4.26+ IFNet checkpoints."""

from __future__ import annotations

import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths
import torch
from comfy import model_management
from comfy_extras.frame_interpolation_models.ifnet import IFNet, detect_rife_config


class RIFEModelLoader:
    """Load a ComfyUI-compatible RIFE IFNet checkpoint and choose inference precision."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("frame_interpolation"),),
                "precision": (["bf16", "fp32_tf32", "fp32_strict"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("INTERP_MODEL",)
    RETURN_NAMES = ("interp_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/video"

    @classmethod
    def load_model(cls, model_name: str, precision: str):
        model_path = folder_paths.get_full_path_or_raise("frame_interpolation", model_name)
        state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
        state_dict = comfy.utils.state_dict_prefix_replace(
            state_dict, {"module.": "", "flownet.": ""}
        )

        key_map: dict[str, str] = {}
        for key in state_dict:
            for index in range(5):
                prefix = f"block{index}."
                if key.startswith(prefix):
                    key_map[key] = f"blocks.{index}.{key[len(prefix) :]}"
        if key_map:
            state_dict = {key_map.get(key, key): value for key, value in state_dict.items()}
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(("teacher.", "caltime."))
        }

        try:
            head_channels, channels = detect_rife_config(state_dict)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Unsupported RIFE checkpoint. Use a ComfyUI-compatible RIFE v4.26+ "
                "checkpoint with the five-block IFNet architecture."
            ) from exc

        model = IFNet(head_ch=head_channels, channels=channels)
        model.load_state_dict(state_dict)

        device = model_management.get_torch_device()
        if precision not in {"bf16", "fp32_tf32", "fp32_strict"}:
            raise ValueError(f"Unsupported RIFE precision: {precision}")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if precision == "bf16" and not model_management.should_use_bf16(device):
            raise RuntimeError(f"BF16 is not supported on the selected ComfyUI device: {device}")

        model.eval().to(dtype=dtype)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        model._rife_allow_tf32 = precision == "fp32_tf32"
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=device,
            offload_device=model_management.unet_offload_device(),
            fast_disk=comfy.storage.state_dict_fast_disk(state_dict),
        )
        return (patcher,)


NODE_CLASS_MAPPINGS = {"RIFEModelLoader": RIFEModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"RIFEModelLoader": "Load RIFE v4.26+ (BF16 / TF32 / FP32)"}
