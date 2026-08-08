import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
import json
from io import BytesIO
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

# ── LoRA adapter registry (2509) ───────────────────────────────────────────────
ADAPTER_SPECS = {
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "anime",
    },
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
    },
    "Light-Restoration": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
        "adapter_name": "light-restoration",
    },
    "Relight": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
        "adapter_name": "relight",
    },
    "Multi-Angle-Lighting": {
        "repo": "dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        "weights": "多角度灯光-251116.safetensors",
        "adapter_name": "multi-angle-lighting",
    },
    "Edit-Skin": {
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
        "adapter_name": "edit-skin",
    },
    "Next-Scene": {
        "repo": "lovis93/next-scene-qwen-image-lora-2509",
        "weights": "next-scene_lora-v2-3000.safetensors",
        "adapter_name": "next-scene",
    },
    "Flat-Log": {
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
        "adapter_name": "flat-log",
    },
    "Upscale-Image": {
        "repo": "vafipas663/Qwen-Edit-2509-Upscale-LoRA",
        "weights": "qwen-edit-enhance_64-v3_000001000.safetensors",
        "adapter_name": "upscale-image",
    },
    "Upscale2K": {
        "repo": "valiantcat/Qwen-Image-Edit-2509-Upscale2K",
        "weights": "qwen_image_edit_2509_upscale.safetensors",
        "adapter_name": "upscale-2k",
    },
    "Dotted-Illustration": {
        "repo": "prithivMLmods/QIE-2509-Dotted-Illustration",
        "weights": "dotted-illustration-2800.safetensors",
        "adapter_name": "dotted-illustration",
    },
}

LOADED_ADAPTERS: set = set()
ADAPTER_NAMES = list(ADAPTER_SPECS.keys())

EXAMPLES_CONFIG = [
    {"image": "examples/1.jpg",   "prompt": "Transform into anime.",                                                                                                                             "lora": "Photo-to-Anime"},
    {"image": "examples/5.jpg",   "prompt": "Remove shadows and relight the image using soft lighting.",                                                                                         "lora": "Light-Restoration"},
    {"image": "examples/4.jpg",   "prompt": "Use a subtle golden-hour filter with smooth light diffusion.",                                                                                      "lora": "Relight"},
    {"image": "examples/2.jpeg",  "prompt": "Rotate the camera 45 degrees to the left.",                                                                                                        "lora": "Multiple-Angles"},
    {"image": "examples/12.jpg",  "prompt": "flatcolor Desaturate the image and lower the contrast to create a flat, ungraded look similar to a camera log profile.",                           "lora": "Flat-Log"},
    {"image": "examples/7.jpg",   "prompt": "Light source from the Right Rear.",                                                                                                                "lora": "Multi-Angle-Lighting"},
    {"image": "examples/10.jpeg", "prompt": "Upscale the image.",                                                                                                                               "lora": "Upscale-Image"},
    {"image": "examples/DI.jpg",  "prompt": "dotted illustration.",                                                                                                                             "lora": "Dotted-Illustration"},
    {"image": "examples/7.jpg",   "prompt": "Light source from the Below.",                                                                                                                     "lora": "Multi-Angle-Lighting"},
    {"image": "examples/2.jpeg",  "prompt": "Switch the camera to a top-down right corner view.",                                                                                               "lora": "Multiple-Angles"},
    {"image": "examples/9.jpg",   "prompt": "The camera moves slightly forward as sunlight breaks through the clouds, casting a soft glow around the character's silhouette in the mist.",      "lora": "Next-Scene"},
    {"image": "examples/8.jpg",   "prompt": "Make the subjects skin details more prominent and natural.",                                                                                        "lora": "Edit-Skin"},
    {"image": "examples/6.jpg",   "prompt": "Switch the camera to a bottom-up view.",                                                                                                           "lora": "Multiple-Angles"},
    {"image": "examples/4.jpg",   "prompt": "Rotate the camera 45 degrees to the right.",                                                                                                      "lora": "Multiple-Angles"},
    {"image": "examples/11.jpg",  "prompt": "Upscale this picture to 4K resolution.",                                                                                                          "lora": "Upscale2K"},
]


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def build_client_config():
    """Static config consumed by the frontend: LoRA list + example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumb": make_thumb_b64(ex["image"]),
            "lora": ex["lora"],
            "prompt": ex["prompt"],
        })
    return {
        "loras": ADAPTER_NAMES,
        "default_lora": "Photo-to-Anime",
        "examples": examples,
    }


print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples and {len(ADAPTER_NAMES)} LoRAs.")


def b64_to_pil(b64_str):
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if b64_str.startswith("data:image"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw = 1024
        nh = int(nw * h / w)
    else:
        nh = 1024
        nw = int(nh * w / h)
    return (nw // 8) * 8, (nh // 8) * 8


# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Qwen-Image-Edit-2509-LoRAs-Fast")


@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    image_b64: str,
    prompt: str,
    lora_adapter: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Edit a single image with Qwen-Image-Edit-2509 + a lazily-loaded LoRA.

    Returns {"image": <base64 PNG data URL>, "seed": <seed used>}.
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_image = b64_to_pil(image_b64)
    if pil_image is None:
        raise gr.Error("Please upload an image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    adapter_name = spec["adapter_name"]
    if adapter_name not in LOADED_ADAPTERS:
        print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
        try:
            pipe.load_lora_weights(spec["repo"], weight_name=spec["weights"], adapter_name=adapter_name)
            LOADED_ADAPTERS.add(adapter_name)
        except Exception as e:
            raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
    else:
        print(f"--- Adapter {lora_adapter} already loaded. ---")

    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )
    width, height = update_dimensions_on_upload(pil_image)

    try:
        result_image = pipe(
            image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example image + prompt + LoRA for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"image": "", "prompt": "", "lora": "", "name": "", "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64 = encode_full_image(ex["image"])
    return {
        "image": b64,
        "prompt": ex["prompt"],
        "lora": ex["lora"],
        "name": os.path.basename(ex["image"]),
        "status": "ok" if b64 else "error",
    }


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: LoRA choices + example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)