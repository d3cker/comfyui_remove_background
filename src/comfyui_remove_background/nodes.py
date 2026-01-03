# ComfyUI custom node: Remove Background (rembg) -> outputs RGB IMAGE + alpha MASK
# deps:
#   pip install rembg onnxruntime pillow
# (for GPU: onnxruntime-gpu, and for Windows preload_dlls may be helpful)

import onnxruntime as ort
import numpy as np
import torch
from PIL import Image

try:
    from rembg import remove, new_session
except Exception as e:
    remove = None
    new_session = None
    _rembg_import_error = e


def _device_choices():
    choices = ["auto", "cpu"]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                choices.append(f"cuda:{i} ({name})")
    except Exception:
        pass
    return choices


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().clamp(0, 1)
    arr = (img.numpy() * 255.0).round().astype(np.uint8)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected image tensor [H,W,3/4], got {arr.shape}")

    if arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB").convert("RGBA")
    return Image.fromarray(arr, mode="RGBA")


def _pil_to_image_and_mask(pil_rgba: Image.Image):
    pil_rgba = pil_rgba.convert("RGBA")
    rgba = np.array(pil_rgba, dtype=np.uint8)

    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    return torch.from_numpy(rgb), torch.from_numpy(alpha)


def _parse_cuda_device_id(device_str: str) -> int:
    # "cuda:0 (RTX ...)" -> 0
    s = device_str.strip()
    s = s.split()[0]          # "cuda:0"
    _, idx = s.split(":")     # "0"
    return int(idx)


class RemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam", 
                        "birefnet-general", "birefnet-general-lite", "birefnet-portrait", "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive"
                     ],
                    {"default": "birefnet-portrait"},
                ),
            },
            "optional": {
                # NOWE:
                "device": (_device_choices(), {"default": "auto"}),
                "allow_cpu_fallback": ("BOOLEAN", {"default": False}),
                "preload_dlls": ("BOOLEAN", {"default": False}),  # may help with Windows

                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"
    CATEGORY = "d3cker/Image"

    def __init__(self):
        self._session = None
        self._session_key = None  # (model, device, allow_cpu_fallback)

    def _get_session(self, model: str, device: str, allow_cpu_fallback: bool, preload_dlls: bool):
        if remove is None or new_session is None:
            raise RuntimeError(
                f"rembg import failed: {_rembg_import_error}\n"
                "Install dependencies in ComfyUI environment:\n"
                "  pip install rembg onnxruntime pillow"
            )

        key = (model, device, bool(allow_cpu_fallback))
        if self._session is not None and self._session_key == key:
            return self._session

        # preload DLLs (onnxruntime-gpu >= 1.21)
        if preload_dlls and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
                ort.print_debug_info() 
            except Exception as e:
                print(f"[rembg] preload_dlls failed: {e}")

        avail = ort.get_available_providers()

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.startswith("cuda:"):
            if "CUDAExecutionProvider" not in avail:
                raise RuntimeError(f"CUDAExecutionProvider not available. available={avail}")
            device_id = _parse_cuda_device_id(device)
            providers = [("CUDAExecutionProvider", {"device_id": device_id})]
            if allow_cpu_fallback:
                providers.append("CPUExecutionProvider")
        else:
            # auto: prefer CUDA if available, then DML, then CPU
            providers = []
            if "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            elif "DmlExecutionProvider" in avail:
                providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")

        print(f"[rembg] available providers: {avail}")
        print(f"[rembg] selected device: {device} | allow_cpu_fallback={allow_cpu_fallback}")
        print(f"[rembg] providers arg: {providers}")

        self._session = new_session(model, providers=providers)
        self._session_key = key

        # rembg BaseSession ma inner_session = ort.InferenceSession(...)
        inner = getattr(self._session, "inner_session", None)
        if inner is not None and hasattr(inner, "get_providers"):
            print(f"[rembg] ONNX session providers in use: {inner.get_providers()}")

        return self._session

    def run(
        self,
        image,
        model="birefnet-portrait",
        device="auto",
        allow_cpu_fallback=True,
        preload_dlls=False,
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    ):
        session = self._get_session(model, device, allow_cpu_fallback, preload_dlls)

        if image.dim() != 4:
            raise ValueError(f"Expected batched IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")

        out_images = []
        out_masks = []

        for i in range(image.shape[0]):
            pil_in = _tensor_to_pil(image[i])

            pil_out = remove(
                pil_in,
                session=session,
                alpha_matting=bool(alpha_matting),
                alpha_matting_foreground_threshold=int(alpha_matting_foreground_threshold),
                alpha_matting_background_threshold=int(alpha_matting_background_threshold),
                alpha_matting_erode_size=int(alpha_matting_erode_size),
            )

            img_t, mask_t = _pil_to_image_and_mask(pil_out)
            out_images.append(img_t)
            out_masks.append(mask_t)

        return (torch.stack(out_images, dim=0), torch.stack(out_masks, dim=0))


NODE_CLASS_MAPPINGS = {"RemoveBackground": RemoveBackground}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoveBackground": "Remove Background (Image + Alpha)"}
