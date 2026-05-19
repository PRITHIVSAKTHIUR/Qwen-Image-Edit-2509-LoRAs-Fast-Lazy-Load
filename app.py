import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
import base64
import json
import html as html_lib
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

# ── LoRA adapter registry ──────────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────
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


def build_example_cards_html():
    cards = ""
    for i, ex in enumerate(EXAMPLES_CONFIG):
        thumb = make_thumb_b64(ex["image"])
        thumb_html = f'<img src="{thumb}" alt="">' if thumb else '<div class="example-thumb-placeholder">Preview</div>'
        lora_badge   = html_lib.escape(ex["lora"])
        prompt_short = html_lib.escape(ex["prompt"][:80])
        if len(ex["prompt"]) > 80:
            prompt_short += "…"
        cards += f'''<div class="example-card" data-idx="{i}">
            <div class="example-thumbs">{thumb_html}</div>
            <div class="example-meta">
                <span class="example-lora-badge">{lora_badge}</span>
            </div>
            <div class="example-prompt-text">{prompt_short}</div>
        </div>'''
    return cards


def load_example_data(idx_str):
    try:
        idx = int(float(idx_str)) if idx_str and idx_str.strip() else -1
    except (ValueError, TypeError):
        idx = -1
    if idx < 0 or idx >= len(EXAMPLES_CONFIG):
        return json.dumps({"image": "", "prompt": "", "lora": "", "status": "error"})
    ex = EXAMPLES_CONFIG[idx]
    b64 = encode_full_image(ex["image"])
    return json.dumps({"image": b64, "prompt": ex["prompt"], "lora": ex["lora"],
                       "name": os.path.basename(ex["image"]), "status": "ok" if b64 else "error"})


print("Building example thumbnails…")
EXAMPLE_CARDS_HTML = build_example_cards_html()
print(f"Built {len(EXAMPLES_CONFIG)} example cards.")


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw, nh = 1024, int(1024 * h / w)
    else:
        nh, nw = 1024, int(1024 * w / h)
    return (nw // 8) * 8, (nh // 8) * 8


def b64_to_pil(b64_str):
    if not b64_str:
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


@spaces.GPU
def infer(image_b64, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps,
          progress=gr.Progress(track_tqdm=True)):
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
        return result_image, seed
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# ── CSS ────────────────────────────────────────────────────────────────────────
css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*{ box-sizing:border-box; margin:0; padding:0; }

:root{
    --pk:       #FF1493;
    --pk-dim:   #CC0074;
    --pk-bright:#FF4DB8;
    --pk-glow:  rgba(255,20,147,.28);
    --pk-soft:  rgba(255,20,147,.12);
    --pk-xsoft: rgba(255,20,147,.06);
    --bg:       #0d0d0f;
    --bg1:      #141416;
    --bg2:      #1a1a1d;
    --bg3:      #222226;
    --border:   #2a2a2e;
    --border2:  #333338;
    --text:     #e8e8ec;
    --text2:    #9898a8;
    --text3:    #555560;
    --mono:     'JetBrains Mono', monospace;
}

html, body, .gradio-container {
    color-scheme: dark !important;
    background: #0d0d0f !important;
}
body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
    color: #e8e8ec !important;
    min-height: 100vh;
    background: #0d0d0f !important;
}
footer { display:none !important; }
.hidden-input { display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important; }

#example-load-btn, #gradio-run-btn {
    position:absolute!important; left:-9999px!important; top:-9999px!important;
    width:1px!important; height:1px!important; opacity:.01!important;
    pointer-events:none!important; overflow:hidden!important;
}

/* ── App shell ── */
.app-shell {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 18px;
    margin: 12px auto;
    max-width: 1440px;
    overflow: hidden;
    box-shadow:
        0 32px 64px -16px rgba(0,0,0,.8),
        0 0 0 1px rgba(255,255,255,.03),
        0 0 60px -20px rgba(255,20,147,.18);
}

/* ── Header ── */
.app-header {
    background: #1a1a1d !important;
    border-bottom: 1px solid #2a2a2e !important;
    padding: 14px 24px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.app-header-left { display:flex; align-items:center; gap:12px; }
.app-logo {
    width:38px; height:38px;
    background: linear-gradient(135deg, #FF1493, #FF4DB8);
    border-radius: 11px;
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 4px 14px rgba(255,20,147,.32);
    flex-shrink:0;
}
.app-logo svg { width:22px; height:22px; fill:#fff; }
.app-title {
    font-size:18px; font-weight:800; letter-spacing:-.4px;
    background: linear-gradient(135deg, #fff, #aaa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-family: 'Inter', sans-serif;
}
.app-badge {
    font-size:11px; font-weight:700; padding:3px 10px;
    border-radius:20px; letter-spacing:.3px;
    background: rgba(255,20,147,.12);
    color: #FF4DB8 !important; -webkit-text-fill-color: #FF4DB8 !important;
    border: 1px solid rgba(255,20,147,.3);
    font-family: 'Inter', sans-serif;
}
.app-badge.fast {
    background: rgba(34,197,94,.1);
    color: #4ade80 !important; -webkit-text-fill-color: #4ade80 !important;
    border: 1px solid rgba(34,197,94,.25);
}

/* ── GitHub button — highlighted DeepPink, always same ── */
.gh-btn {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding: 7px 16px !important;
    border-radius: 8px !important;
    text-decoration: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: .1px !important;
    background: #FF1493 !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    box-shadow: 0 2px 10px rgba(255,20,147,.45), 0 1px 0 rgba(255,255,255,.1) inset !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
    cursor: pointer !important; flex-shrink:0 !important;
}
.gh-btn:hover {
    background: #FF4DB8 !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(255,20,147,.6), 0 1px 0 rgba(255,255,255,.12) inset !important;
}
.gh-btn:active {
    background: #CC0074 !important;
    transform: translateY(0) !important;
    box-shadow: 0 1px 5px rgba(255,20,147,.35) !important;
}
.gh-btn svg { fill:#ffffff !important; flex-shrink:0; width:15px!important; height:15px!important; }
.gh-btn span { color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; }

@media (prefers-color-scheme: light) {
    .gh-btn { background:#FF1493!important; color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; }
    .gh-btn:hover { background:#FF4DB8!important; }
    .gh-btn svg { fill:#ffffff!important; }
    .gh-btn span { color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; }
}
@media (prefers-color-scheme: dark) {
    .gh-btn { background:#FF1493!important; color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; }
    .gh-btn:hover { background:#FF4DB8!important; }
    .gh-btn svg { fill:#ffffff!important; }
    .gh-btn span { color:#ffffff!important; -webkit-text-fill-color:#ffffff!important; }
}

/* ── Toolbar ── */
.app-toolbar {
    background: #141416 !important;
    border-bottom: 1px solid #2a2a2e !important;
    padding: 7px 16px;
    display:flex; gap:4px; align-items:center; flex-wrap:wrap;
}
.tb-sep { width:1px; height:28px; background:#2a2a2e; margin:0 8px; }
.modern-tb-btn {
    display:inline-flex; align-items:center; justify-content:center; gap:6px;
    min-width:32px; height:34px; background:transparent; border:1px solid transparent;
    border-radius:8px; cursor:pointer; font-size:13px; font-weight:700; padding:0 12px;
    font-family:'Inter',sans-serif;
    color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
    transition:all .15s ease;
}
.modern-tb-btn:hover { background:rgba(255,20,147,.12); border-color:rgba(255,20,147,.4); }
.modern-tb-btn:active { background:rgba(255,20,147,.22); border-color:#FF1493; }
.modern-tb-btn .tb-label { font-size:13px; color:#fff!important; -webkit-text-fill-color:#fff!important; font-weight:700; }
.modern-tb-btn .tb-svg { width:15px; height:15px; flex-shrink:0; }
.modern-tb-btn .tb-svg, .modern-tb-btn .tb-svg * { stroke:#fff!important; fill:none!important; }
.tb-info { font-family:var(--mono); font-size:12px; color:#555560; padding:0 8px; display:flex; align-items:center; }

@media (prefers-color-scheme: light) {
    .modern-tb-btn { color:#fff!important; -webkit-text-fill-color:#fff!important; }
    .modern-tb-btn .tb-label { color:#fff!important; -webkit-text-fill-color:#fff!important; }
    .modern-tb-btn .tb-svg, .modern-tb-btn .tb-svg * { stroke:#fff!important; }
}

/* ── Main layout ── */
.app-main-row { display:flex; gap:0; flex:1; overflow:hidden; }
.app-main-left { flex:1; display:flex; flex-direction:column; min-width:0; border-right:1px solid #2a2a2e; }
.app-main-right { width:440px; display:flex; flex-direction:column; flex-shrink:0; background:#141416!important; }

/* ── Drop zone ── */
#gallery-drop-zone { position:relative; background:#09090b!important; min-height:440px; overflow:auto; }
#gallery-drop-zone.drag-over { outline:2px solid #FF1493; outline-offset:-2px; background:rgba(255,20,147,.05)!important; }

.upload-prompt-modern { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); z-index:20; }
.upload-click-area {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    cursor:pointer; padding:36px 52px; border:2px dashed #333338; border-radius:16px;
    background:rgba(255,20,147,.05); transition:all .2s ease; gap:8px;
}
.upload-click-area:hover { background:rgba(255,20,147,.1); border-color:#FF1493; transform:scale(1.03); }
.upload-click-area:active { background:rgba(255,20,147,.15); transform:scale(.98); }
.upload-click-area svg { width:80px; height:80px; }
.upload-main-text { color:#9898a8; font-size:14px; font-weight:600; margin-top:4px; font-family:'Inter',sans-serif; }
.upload-sub-text { color:#555560; font-size:12px; font-weight:400; text-align:center; max-width:280px; line-height:1.5; font-family:'Inter',sans-serif; }

/* ── Gallery grid ── */
.image-gallery-grid {
    display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
    gap:12px; padding:16px; align-content:start;
}
.gallery-thumb {
    position:relative; aspect-ratio:1; border-radius:10px; overflow:hidden;
    cursor:pointer; border:2px solid #2a2a2e; transition:all .2s ease; background:#1a1a1d;
}
.gallery-thumb:hover { border-color:#333338; transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.4); }
.gallery-thumb.selected { border-color:#FF1493!important; box-shadow:0 0 0 3px rgba(255,20,147,.25); }
.gallery-thumb img { width:100%; height:100%; object-fit:cover; }
.thumb-badge {
    position:absolute; top:6px; left:6px; background:#FF1493; color:#fff;
    padding:2px 8px; border-radius:4px; font-family:var(--mono); font-size:11px; font-weight:600;
}
.thumb-remove {
    position:absolute; top:6px; right:6px; width:24px; height:24px;
    background:rgba(0,0,0,.75); color:#fff; border:1px solid rgba(255,255,255,.15);
    border-radius:50%; cursor:pointer; display:none;
    align-items:center; justify-content:center; font-size:12px; transition:all .15s; line-height:1;
}
.gallery-thumb:hover .thumb-remove { display:flex; }
.thumb-remove:hover { background:#FF1493; border-color:#FF1493; }

/* ── Hint bar ── */
.hint-bar {
    background: rgba(255,20,147,.05) !important;
    border-top:1px solid #2a2a2e; border-bottom:1px solid #2a2a2e;
    padding:10px 20px; font-size:13px; color:#9898a8; line-height:1.7;
    font-weight:400; font-family:'Inter',sans-serif;
}
.hint-bar b { color:#FF4DB8; font-weight:700; }
.hint-bar kbd {
    display:inline-block; padding:1px 6px; background:#222226;
    border:1px solid #333338; border-radius:4px;
    font-family:var(--mono); font-size:11px; color:#9898a8;
}

/* ── Suggestions ── */
.suggestions-section { border-top:1px solid #2a2a2e; padding:12px 16px; background:#141416!important; }
.suggestions-title, .examples-title {
    font-size:11px; font-weight:700; color:#555560;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;
    font-family:'Inter',sans-serif;
}
.suggestions-wrap { display:flex; flex-wrap:wrap; gap:6px; }
.suggestion-chip {
    display:inline-flex; align-items:center; gap:4px; padding:5px 13px;
    background:rgba(255,20,147,.1); border:1px solid rgba(255,20,147,.25); border-radius:20px;
    color:#FF4DB8; font-size:12px; font-weight:600; font-family:'Inter',sans-serif;
    cursor:pointer; transition:all .15s; white-space:nowrap;
}
.suggestion-chip:hover { background:rgba(255,20,147,.2); border-color:#FF1493; color:#fff; transform:translateY(-1px); }

/* ── Examples ── */
.examples-section { border-top:1px solid #2a2a2e; padding:14px 16px 18px; background:#141416!important; }
.examples-scroll { display:flex; gap:10px; overflow-x:auto; padding-bottom:10px; padding-top:2px; }
.examples-scroll::-webkit-scrollbar { height:5px; }
.examples-scroll::-webkit-scrollbar-track { background:#0d0d0f; border-radius:3px; }
.examples-scroll::-webkit-scrollbar-thumb { background:#333338; border-radius:3px; }
.examples-scroll::-webkit-scrollbar-thumb:hover { background:#CC0074; }
.example-card {
    flex-shrink:0; width:200px; background:#1a1a1d!important; border:1px solid #2a2a2e;
    border-radius:12px; overflow:hidden; cursor:pointer; transition:all .2s ease;
}
.example-card:hover { border-color:#FF1493; transform:translateY(-3px); box-shadow:0 6px 20px rgba(255,20,147,.22); }
.example-card.loading { opacity:.5; pointer-events:none; }
.example-thumbs { display:flex; height:115px; overflow:hidden; background:#222226!important; }
.example-thumbs img { flex:1; object-fit:cover; min-width:0; width:100%; }
.example-thumb-placeholder {
    flex:1; display:flex; align-items:center; justify-content:center;
    background:#222226!important; color:#555560; font-size:11px;
}
.example-meta { padding:6px 10px 3px; display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
.example-lora-badge {
    display:inline-flex; padding:2px 7px; background:rgba(255,20,147,.1); border-radius:4px;
    font-size:10px; font-weight:700; color:#FF4DB8; font-family:var(--mono);
    white-space:nowrap; border:1px solid rgba(255,20,147,.2);
}
.example-prompt-text {
    padding:2px 10px 10px; font-size:11.5px; color:#9898a8; line-height:1.45;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; font-weight:400; font-family:'Inter',sans-serif;
}

/* ── Right panel ── */
.panel-card { border-bottom:1px solid #2a2a2e; background:#141416!important; }
.panel-card-title {
    padding:11px 20px; font-size:11px; font-weight:700; color:#555560;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid rgba(42,42,46,.6);
    background:#141416!important; font-family:'Inter',sans-serif;
}
.panel-card-body { padding:14px 18px; display:flex; flex-direction:column; gap:8px; background:#141416!important; }
.modern-label { font-size:13px; font-weight:600; color:#9898a8; margin-bottom:4px; display:block; font-family:'Inter',sans-serif; }
.modern-textarea {
    width:100%; background:#09090b!important; border:1px solid #2a2a2e; border-radius:8px;
    padding:10px 14px; font-family:'Inter',sans-serif; font-size:14px;
    color:#e8e8ec!important; -webkit-text-fill-color:#e8e8ec!important;
    resize:vertical; outline:none; min-height:44px; transition:border-color .2s; font-weight:400;
}
.modern-textarea:focus { border-color:#FF1493; box-shadow:0 0 0 3px rgba(255,20,147,.22); }
.modern-textarea::placeholder { color:#555560!important; -webkit-text-fill-color:#555560!important; }
.modern-textarea.error-flash {
    border-color:#ef4444!important; box-shadow:0 0 0 3px rgba(239,68,68,.2)!important;
    animation:shake .4s ease;
}
@keyframes shake {
    0%,100%{transform:translateX(0)} 20%,60%{transform:translateX(-4px)} 40%,80%{transform:translateX(4px)}
}

/* ── LoRA selector - always dark ── */
.lora-selector-card { border-bottom:1px solid #2a2a2e!important; background:#0d0d0f!important; }
.lora-selector-body { padding:12px 18px!important; background:#0d0d0f!important; }
.lora-select-label {
    font-size:11px!important; font-weight:700!important;
    color:#555560!important; -webkit-text-fill-color:#555560!important;
    text-transform:uppercase!important; letter-spacing:1px!important;
    margin-bottom:8px!important; display:flex!important; align-items:center!important;
    gap:6px!important; font-family:'Inter',sans-serif!important;
}
.lora-select-label::before {
    content:''; display:inline-block; width:8px; height:8px;
    background:#FF1493; border-radius:50%; flex-shrink:0;
}
.lora-native-select {
    width:100%!important; background:#09090b!important;
    border:1px solid #333338!important; border-radius:8px!important;
    padding:9px 36px 9px 14px!important;
    font-family:'Inter',sans-serif!important; font-size:13px!important; font-weight:600!important;
    color:#e8e8ec!important; -webkit-text-fill-color:#e8e8ec!important;
    outline:none!important; appearance:none!important; -webkit-appearance:none!important;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23FF1493' d='M6 8L1 3h10z'/%3E%3C/svg%3E")!important;
    background-repeat:no-repeat!important; background-position:right 12px center!important;
    cursor:pointer!important; transition:border-color .2s!important; color-scheme:dark!important;
}
.lora-native-select:focus { border-color:#FF1493!important; box-shadow:0 0 0 3px rgba(255,20,147,.22)!important; }
.lora-native-select option { background:#1a1a1d!important; color:#e8e8ec!important; }

@media (prefers-color-scheme: light) {
    .lora-selector-card { background:#0d0d0f!important; }
    .lora-selector-body { background:#0d0d0f!important; }
    .lora-select-label { color:#555560!important; -webkit-text-fill-color:#555560!important; }
    .lora-native-select {
        background:#09090b!important; color:#e8e8ec!important;
        -webkit-text-fill-color:#e8e8ec!important; border-color:#333338!important; color-scheme:dark!important;
    }
    .lora-native-select option { background:#1a1a1d!important; color:#e8e8ec!important; }
}

/* ── Toast ── */
.toast-notification {
    position:fixed; top:24px; left:50%; transform:translateX(-50%) translateY(-120%);
    z-index:9999; padding:10px 24px; border-radius:10px;
    font-family:'Inter',sans-serif; font-size:14px; font-weight:700;
    display:flex; align-items:center; gap:8px; box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;
    opacity:0; pointer-events:none;
}
.toast-notification.visible { transform:translateX(-50%) translateY(0); opacity:1; pointer-events:auto; }
.toast-notification.error   { background:linear-gradient(135deg,#dc2626,#b91c1c); color:#fff; border:1px solid rgba(255,255,255,.15); }
.toast-notification.warning { background:linear-gradient(135deg,#FF1493,#CC0074); color:#fff; border:1px solid rgba(255,255,255,.15); }
.toast-notification.info    { background:linear-gradient(135deg,#2563eb,#1d4ed8); color:#fff; border:1px solid rgba(255,255,255,.15); }

/* ── Run button ── */
.btn-run {
    display:flex; align-items:center; justify-content:center; gap:8px; width:100%;
    background:linear-gradient(135deg,#FF1493,#CC0074); border:none; border-radius:10px;
    padding:13px 24px; cursor:pointer; font-size:15px; font-weight:800;
    font-family:'Inter',sans-serif;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
    transition:all .2s ease; letter-spacing:.2px;
    box-shadow:0 4px 20px rgba(255,20,147,.3),inset 0 1px 0 rgba(255,255,255,.12);
}
.btn-run:hover {
    background:linear-gradient(135deg,#FF4DB8,#FF1493); transform:translateY(-1px);
    box-shadow:0 8px 28px rgba(255,20,147,.5),inset 0 1px 0 rgba(255,255,255,.15);
}
.btn-run:active { transform:translateY(0); box-shadow:0 2px 10px rgba(255,20,147,.3); }
.btn-run svg { width:18px; height:18px; fill:#fff!important; }
#custom-run-btn, #custom-run-btn *, #run-btn-label, .btn-run, .btn-run * {
    color:#fff!important; -webkit-text-fill-color:#fff!important; fill:#fff!important;
}

/* ── Output ── */
.output-frame { border-bottom:1px solid #2a2a2e; display:flex; flex-direction:column; position:relative; }
.out-title {
    padding:10px 20px; font-size:11px; font-weight:700;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid rgba(42,42,46,.6);
    display:flex; align-items:center; justify-content:space-between;
    background:#141416!important; font-family:'Inter',sans-serif;
}
.out-body {
    flex:1; background:#09090b!important;
    display:flex; align-items:center; justify-content:center;
    overflow:hidden; min-height:260px; position:relative;
}
.out-body img { max-width:100%; max-height:480px; image-rendering:auto; }
.out-placeholder { color:#555560; font-size:13px; text-align:center; padding:20px; font-weight:500; font-family:'Inter',sans-serif; }
.out-download-btn {
    display:none; align-items:center; justify-content:center;
    background:rgba(255,20,147,.12); border:1px solid rgba(255,20,147,.28); border-radius:6px;
    cursor:pointer; padding:3px 10px; font-size:11px; font-weight:700;
    color:#FF4DB8!important; -webkit-text-fill-color:#FF4DB8!important;
    gap:4px; height:24px; transition:all .15s; font-family:'Inter',sans-serif;
}
.out-download-btn:hover {
    background:#FF1493; border-color:#FF1493;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
}
.out-download-btn.visible { display:inline-flex; }
.out-download-btn svg { width:12px; height:12px; fill:#FF4DB8; }
.out-download-btn:hover svg { fill:#fff; }

/* ── Loader ── */
.modern-loader {
    display:none; position:absolute; top:0; left:0; right:0; bottom:0;
    background:rgba(9,9,11,.93); z-index:15;
    flex-direction:column; align-items:center; justify-content:center;
    gap:16px; backdrop-filter:blur(4px);
}
.modern-loader.active { display:flex; }
.modern-loader .loader-spinner {
    width:36px; height:36px; border:3px solid #2a2a2e;
    border-top-color:#FF1493; border-radius:50%; animation:spin .8s linear infinite;
}
@keyframes spin { to{transform:rotate(360deg)} }
.modern-loader .loader-text { font-size:13px; color:#9898a8; font-weight:600; font-family:'Inter',sans-serif; }
.loader-bar-track { width:200px; height:4px; background:#2a2a2e; border-radius:2px; overflow:hidden; }
.loader-bar-fill {
    height:100%; background:linear-gradient(90deg,#FF1493,#FF4DB8,#FF1493);
    background-size:200% 100%; animation:shimmer 1.5s ease-in-out infinite; border-radius:2px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── Settings ── */
.settings-group { border:1px solid #2a2a2e; border-radius:10px; margin:12px 16px; padding:0; overflow:hidden; background:#141416!important; }
.settings-group-title {
    font-size:11px; font-weight:700; color:#555560; text-transform:uppercase; letter-spacing:1px;
    padding:9px 16px; border-bottom:1px solid #2a2a2e;
    background:rgba(26,26,29,.5)!important; font-family:'Inter',sans-serif;
}
.settings-group-body { padding:14px 16px; display:flex; flex-direction:column; gap:12px; background:#141416!important; }
.slider-row { display:flex; align-items:center; gap:10px; min-height:28px; }
.slider-row label { font-size:13px; font-weight:600; color:#9898a8; min-width:72px; flex-shrink:0; font-family:'Inter',sans-serif; }
.slider-row input[type="range"] {
    flex:1; -webkit-appearance:none; appearance:none;
    height:5px; background:#333338; border-radius:3px; outline:none; min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance:none; width:16px; height:16px;
    background:linear-gradient(135deg,#FF1493,#CC0074);
    border-radius:50%; cursor:pointer;
    box-shadow:0 2px 6px rgba(255,20,147,.3); transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover { transform:scale(1.2); }
.slider-row input[type="range"]::-moz-range-thumb {
    width:16px; height:16px; background:linear-gradient(135deg,#FF1493,#CC0074);
    border-radius:50%; cursor:pointer; border:none;
}
.slider-row .slider-val {
    min-width:52px; text-align:right; font-family:var(--mono); font-size:12px;
    font-weight:500; padding:3px 8px; background:#09090b; border:1px solid #2a2a2e;
    border-radius:6px; color:#9898a8; flex-shrink:0;
}
.checkbox-row { display:flex; align-items:center; gap:8px; font-size:13px; color:#9898a8; }
.checkbox-row input[type="checkbox"] { accent-color:#FF1493; width:16px; height:16px; cursor:pointer; }
.checkbox-row label { color:#9898a8; font-size:13px; cursor:pointer; font-weight:500; font-family:'Inter',sans-serif; }

/* ── Status bar ── */
.app-statusbar {
    background:#141416!important; border-top:1px solid #2a2a2e;
    padding:6px 20px; display:flex; gap:12px; height:34px; align-items:center;
}
.app-statusbar .sb-section {
    padding:0 12px; flex:1; display:flex; align-items:center;
    font-family:var(--mono); font-size:12px; color:#555560;
    overflow:hidden; white-space:nowrap;
}
.app-statusbar .sb-section.sb-fixed {
    flex:0 0 auto; min-width:90px; text-align:center; justify-content:center;
    padding:3px 12px; background:rgba(255,20,147,.1); border-radius:6px;
    color:#FF4DB8; font-weight:700; border:1px solid rgba(255,20,147,.2);
}

/* ── Footer ── */
.exp-note {
    padding:10px 20px; font-size:12px; color:#555560;
    border-top:1px solid #2a2a2e; text-align:center; font-weight:500;
    background:#141416!important; font-family:'Inter',sans-serif;
}
.exp-note a { color:#FF4DB8!important; -webkit-text-fill-color:#FF4DB8!important; text-decoration:none; }
.exp-note a:hover { text-decoration:underline; color:#fff!important; -webkit-text-fill-color:#fff!important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width:7px; height:7px; }
::-webkit-scrollbar-track { background:#09090b; }
::-webkit-scrollbar-thumb { background:#333338; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#CC0074; }

/* ── Responsive ── */
@media(max-width:860px){
    .app-main-row { flex-direction:column; }
    .app-main-right { width:100%; }
    .app-main-left { border-right:none; border-bottom:1px solid #2a2a2e; }
}
"""

# ── JavaScript ─────────────────────────────────────────────────────────────────
gallery_js = r"""
() => {
function init() {
    if (window.__qwenInitDone) return;

    const dropZone     = document.getElementById('gallery-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick  = document.getElementById('upload-click-area');
    const fileInput    = document.getElementById('custom-file-input');
    const imgPreview   = document.getElementById('single-img-preview');
    const imgContainer = document.getElementById('single-img-container');
    const btnUpload    = document.getElementById('tb-upload');
    const btnClear     = document.getElementById('tb-clear');
    const promptInput  = document.getElementById('custom-prompt-input');
    const loraSelect   = document.getElementById('custom-lora-select');
    const runBtnEl     = document.getElementById('custom-run-btn');
    const imgCountTb   = document.getElementById('tb-image-count');
    const imgCountSb   = document.getElementById('sb-image-count');

    if (!dropZone || !fileInput) { setTimeout(init, 250); return; }
    window.__qwenInitDone = true;

    let currentImage = null;
    let toastTimer = null;

    /* ── Force dark styles ── */
    function enforceDark() {
        const loraCard  = document.querySelector('.lora-selector-card');
        const loraBody  = document.querySelector('.lora-selector-body');
        const loraLabel = document.querySelector('.lora-select-label');
        const loraEl    = document.getElementById('custom-lora-select');
        if (loraCard)  loraCard.style.setProperty('background','#0d0d0f','important');
        if (loraBody)  loraBody.style.setProperty('background','#0d0d0f','important');
        if (loraLabel) { loraLabel.style.setProperty('color','#555560','important'); loraLabel.style.setProperty('-webkit-text-fill-color','#555560','important'); }
        if (loraEl)    { loraEl.style.setProperty('background-color','#09090b','important'); loraEl.style.setProperty('color','#e8e8ec','important'); loraEl.style.setProperty('-webkit-text-fill-color','#e8e8ec','important'); loraEl.style.setProperty('border-color','#333338','important'); }
        const ghBtn = document.querySelector('.gh-btn');
        if (ghBtn) {
            ghBtn.style.setProperty('background','#FF1493','important');
            ghBtn.style.setProperty('color','#ffffff','important');
            ghBtn.style.setProperty('-webkit-text-fill-color','#ffffff','important');
            const svg = ghBtn.querySelector('svg'); if (svg) svg.style.setProperty('fill','#ffffff','important');
            const span = ghBtn.querySelector('span'); if (span) { span.style.setProperty('color','#ffffff','important'); span.style.setProperty('-webkit-text-fill-color','#ffffff','important'); }
        }
        const shell  = document.querySelector('.app-shell');
        const header = document.querySelector('.app-header');
        const bar    = document.querySelector('.app-toolbar');
        if (shell)  shell.style.setProperty('background','#141416','important');
        if (header) header.style.setProperty('background','#1a1a1d','important');
        if (bar)    bar.style.setProperty('background','#141416','important');
    }
    enforceDark();
    setInterval(enforceDark, 800);

    /* GitHub hover */
    const ghBtn = document.querySelector('.gh-btn');
    if (ghBtn) {
        ghBtn.addEventListener('mouseenter', () => { ghBtn.style.setProperty('background','#FF4DB8','important'); ghBtn.style.setProperty('transform','translateY(-1px)','important'); ghBtn.style.setProperty('box-shadow','0 5px 18px rgba(255,20,147,.6)','important'); });
        ghBtn.addEventListener('mouseleave', () => { ghBtn.style.setProperty('background','#FF1493','important'); ghBtn.style.setProperty('transform','translateY(0)','important'); ghBtn.style.setProperty('box-shadow','0 2px 10px rgba(255,20,147,.45)','important'); });
        ghBtn.addEventListener('mousedown',  () => { ghBtn.style.setProperty('background','#CC0074','important'); });
        ghBtn.addEventListener('mouseup',    () => { ghBtn.style.setProperty('background','#FF4DB8','important'); });
    }

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div'); toast.id = 'app-toast'; toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon'); const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        icon.textContent = type === 'warning' ? '\u26A0' : type === 'info' ? '\u2139' : '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth; toast.classList.add('visible');
        toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
    }
    window.__showToast = showToast;

    function flashPromptError() {
        if (!promptInput) return;
        promptInput.classList.add('error-flash'); promptInput.focus();
        setTimeout(() => promptInput.classList.remove('error-flash'), 800);
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId); if (!container) return;
        container.querySelectorAll('input, textarea').forEach(el => {
            if (el.type === 'file' || el.type === 'range' || el.type === 'checkbox') return;
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) { ns.set.call(el, value); el.dispatchEvent(new Event('input',{bubbles:true,composed:true})); el.dispatchEvent(new Event('change',{bubbles:true,composed:true})); }
        });
    }
    window.__setGradioValue = setGradioValue;

    function updateCount(hasImg) {
        const txt = hasImg ? '1 image' : 'No images';
        if (imgCountTb) imgCountTb.textContent = txt;
        if (imgCountSb) imgCountSb.textContent = hasImg ? '1 image uploaded' : 'No images uploaded';
    }

    function setImage(b64, name) {
        currentImage = {b64, name};
        if (imgPreview) { imgPreview.src = b64; imgPreview.style.display = 'block'; }
        if (imgContainer) imgContainer.style.display = 'block';
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        setGradioValue('hidden-image-b64', b64);
        updateCount(true);
    }
    window.__setImage = setImage;

    function clearImage() {
        currentImage = null;
        if (imgPreview) { imgPreview.src = ''; imgPreview.style.display = 'none'; }
        if (imgContainer) imgContainer.style.display = 'none';
        if (uploadPrompt) uploadPrompt.style.display = '';
        setGradioValue('hidden-image-b64', '');
        updateCount(false);
    }
    window.__clearImage = clearImage;

    function syncPrompt() { if (promptInput) setGradioValue('prompt-gradio-input', promptInput.value); }
    function syncLora() {
        if (!loraSelect) return;
        const c = document.getElementById('gradio-lora'); if (!c) return;
        c.querySelectorAll('input').forEach(el => {
            const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value');
            if (ns && ns.set) { ns.set.call(el,loraSelect.value); el.dispatchEvent(new Event('input',{bubbles:true,composed:true})); el.dispatchEvent(new Event('change',{bubbles:true,composed:true})); }
        });
    }

    function processFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => setImage(e.target.result, file.name);
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => { if (e.target.files[0]) processFile(e.target.files[0]); e.target.value = ''; });
    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload)   btnUpload.addEventListener('click',  () => fileInput.click());
    if (btnClear)    btnClear.addEventListener('click',   clearImage);

    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
    });

    /* Click on preview to replace */
    if (imgContainer) imgContainer.addEventListener('click', () => fileInput.click());

    if (promptInput) promptInput.addEventListener('input', syncPrompt);
    if (loraSelect)  loraSelect.addEventListener('change', syncLora);

    window.__setPrompt = function(text) { if (promptInput) { promptInput.value = text; syncPrompt(); } };
    window.__setLora   = function(lora) {
        if (loraSelect) { loraSelect.value = lora; loraSelect.dispatchEvent(new Event('change',{bubbles:true})); syncLora(); }
    };

    /* Example card clicks */
    document.querySelectorAll('.example-card[data-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = card.getAttribute('data-idx');
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            card.classList.add('loading'); showToast('Loading example\u2026','info');
            setGradioValue('example-result-data',''); setGradioValue('example-idx-input', idx);
            setTimeout(() => { const btn = document.getElementById('example-load-btn'); if (btn) { const b = btn.querySelector('button'); if (b) b.click(); else btn.click(); } }, 150);
            setTimeout(() => card.classList.remove('loading'), 12000);
        });
    });

    /* Sliders */
    function syncSlider(customId, gradioId) {
        const slider = document.getElementById(customId); const valSpan = document.getElementById(customId+'-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const c = document.getElementById(gradioId); if (!c) return;
            c.querySelectorAll('input[type="range"],input[type="number"]').forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value');
                if (ns && ns.set) { ns.set.call(el,slider.value); el.dispatchEvent(new Event('input',{bubbles:true,composed:true})); el.dispatchEvent(new Event('change',{bubbles:true,composed:true})); }
            });
        });
    }
    syncSlider('custom-seed','gradio-seed'); syncSlider('custom-guidance','gradio-guidance'); syncSlider('custom-steps','gradio-steps');

    const randCheck = document.getElementById('custom-randomize');
    if (randCheck) {
        randCheck.addEventListener('change', () => {
            const c = document.getElementById('gradio-randomize'); if (!c) return;
            const cb = c.querySelector('input[type="checkbox"]'); if (cb && cb.checked !== randCheck.checked) cb.click();
        });
    }

    function showLoader() { const l=document.getElementById('output-loader'); if(l) l.classList.add('active'); const sb=document.querySelector('.sb-fixed'); if(sb) sb.textContent='Processing\u2026'; }
    function hideLoader() { const l=document.getElementById('output-loader'); if(l) l.classList.remove('active'); const sb=document.querySelector('.sb-fixed'); if(sb) sb.textContent='Done'; }
    window.__showLoader = showLoader; window.__hideLoader = hideLoader;

    function validateAndRun() {
        const pv = promptInput ? promptInput.value.trim() : '';
        if (!currentImage && !pv) { showToast('Please upload an image and enter a prompt','error'); flashPromptError(); return false; }
        if (!currentImage) { showToast('Please upload an image','error'); return false; }
        if (!pv) { showToast('Please enter an edit prompt','warning'); flashPromptError(); return false; }
        return true;
    }

    window.__clickRun = function() {
        if (!validateAndRun()) return;
        syncPrompt(); syncLora();
        setGradioValue('hidden-image-b64', currentImage ? currentImage.b64 : '');
        showLoader();
        setTimeout(() => { const gb = document.getElementById('gradio-run-btn'); if (!gb) return; const b = gb.querySelector('button'); if (b) b.click(); else gb.click(); }, 200);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', () => window.__clickRun());
    updateCount(false);
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs() {
    const resultContainer = document.getElementById('gradio-result');
    const outBody = document.getElementById('output-image-container');
    const outPh   = document.getElementById('output-placeholder');
    const dlBtn   = document.getElementById('dl-btn-output');
    if (!resultContainer || !outBody) { setTimeout(watchOutputs, 500); return; }
    if (dlBtn) {
        dlBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const img = outBody.querySelector('img.modern-out-img');
            if (img && img.src) { const a=document.createElement('a'); a.href=img.src; a.download='qwen_edit_output.png'; document.body.appendChild(a); a.click(); document.body.removeChild(a); }
        });
    }
    function syncImage() {
        const ri = resultContainer.querySelector('img');
        if (ri && ri.src) {
            if (outPh) outPh.style.display='none';
            let ex = outBody.querySelector('img.modern-out-img');
            if (!ex) { ex=document.createElement('img'); ex.className='modern-out-img'; outBody.appendChild(ex); }
            if (ex.src !== ri.src) { ex.src=ri.src; if(dlBtn) dlBtn.classList.add('visible'); if(window.__hideLoader) window.__hideLoader(); }
        }
    }
    const obs = new MutationObserver(syncImage);
    obs.observe(resultContainer,{childList:true,subtree:true,attributes:true,attributeFilter:['src']});
    setInterval(syncImage, 800);
}
watchOutputs();

function watchSeed() {
    const sc=document.getElementById('gradio-seed'); const sl=document.getElementById('custom-seed'); const sv=document.getElementById('custom-seed-val');
    if (!sc||!sl) { setTimeout(watchSeed,500); return; }
    function sync() { const el=sc.querySelector('input[type="range"],input[type="number"]'); if(el&&el.value){sl.value=el.value;if(sv)sv.textContent=el.value;} }
    const obs=new MutationObserver(sync); obs.observe(sc,{childList:true,subtree:true,attributes:true,attributeFilter:['value']}); setInterval(sync,1000);
}
watchSeed();

function watchExamples() {
    const container = document.getElementById('example-result-data');
    if (!container) { setTimeout(watchExamples,500); return; }
    let last = '';
    function check() {
        const el = container.querySelector('textarea')||container.querySelector('input'); if (!el) return;
        const val = el.value; if (!val||val===last||val.length<10) return;
        try {
            const data = JSON.parse(val);
            if (data.status==='ok' && data.image) {
                last = val;
                if (window.__clearImage) window.__clearImage();
                if (window.__setImage)   window.__setImage(data.image, data.name||'example.jpg');
                if (window.__setPrompt && data.prompt) window.__setPrompt(data.prompt);
                if (window.__setLora   && data.lora)   window.__setLora(data.lora);
                document.querySelectorAll('.example-card.loading').forEach(c=>c.classList.remove('loading'));
                if (window.__showToast) window.__showToast('Example loaded','info');
            } else if (data.status==='error') {
                document.querySelectorAll('.example-card.loading').forEach(c=>c.classList.remove('loading'));
                if (window.__showToast) window.__showToast('Could not load example','error');
            }
        } catch(e) { console.error('Example parse error:',e); }
    }
    const obs=new MutationObserver(check); obs.observe(container,{childList:true,subtree:true,characterData:true,attributes:true}); setInterval(check,500);
}
watchExamples();
}
"""

# ── SVG assets ─────────────────────────────────────────────────────────────────
LOGO_SVG = '''<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2C12 2 7 6.5 7 13c0 1.4.3 2.7.8 3.9L5 19.7l1.4 1.4 2.8-2.8c1.2.5 2.5.7 3.8.7s2.6-.2 3.8-.7l2.8 2.8 1.4-1.4-2.8-2.8c.5-1.2.8-2.5.8-3.9 0-6.5-5-11-5-11z"/>
  <circle cx="12" cy="13" r="2"/>
  <path d="M9 21c0 1.1.9 2 2 2h2c1.1 0 2-.9 2-2v-1H9v1z"/>
</svg>'''

UPLOAD_SVG   = '<svg class="tb-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
CLEAR_SVG    = '<svg class="tb-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>'
DOWNLOAD_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-5-5h3V4h4v7h3l-5 5z" fill="currentColor"/><path d="M20 18H4v2h16v-2z" fill="currentColor"/></svg>'
GITHUB_SVG   = '<svg width="15" height="15" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path fill="#ffffff" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'

LORA_OPTIONS_HTML = "\n".join(
    f'<option value="{html_lib.escape(n)}">{html_lib.escape(n)}</option>'
    for n in ADAPTER_NAMES
)

# ── Gradio app ─────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:

    # Hidden state
    hidden_image_b64 = gr.Textbox(value="",    elem_id="hidden-image-b64",    elem_classes="hidden-input", container=False)
    prompt           = gr.Textbox(value="",    elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False)
    lora_adapter     = gr.Dropdown(choices=ADAPTER_NAMES, value="Photo-to-Anime", elem_id="gradio-lora", elem_classes="hidden-input", container=False)
    seed             = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, value=0, elem_id="gradio-seed",     elem_classes="hidden-input", container=False)
    randomize_seed   = gr.Checkbox(value=True, elem_id="gradio-randomize",    elem_classes="hidden-input", container=False)
    guidance_scale   = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, elem_id="gradio-guidance", elem_classes="hidden-input", container=False)
    steps            = gr.Slider(minimum=1, maximum=50, step=1, value=4,      elem_id="gradio-steps",    elem_classes="hidden-input", container=False)
    result           = gr.Image(elem_id="gradio-result",                      elem_classes="hidden-input", container=False, format="png")

    example_idx      = gr.Textbox(value="", elem_id="example-idx-input",   elem_classes="hidden-input", container=False)
    example_result   = gr.Textbox(value="", elem_id="example-result-data", elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">

      <!-- Header -->
      <div class="app-header">
        <div class="app-header-left">
          <div class="app-logo">{LOGO_SVG}</div>
          <span class="app-title">Qwen-Image-Edit</span>
          <span class="app-badge">2509</span>
          <span class="app-badge fast">4-Step Fast</span>
        </div>
        <a href="https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load"
           target="_blank" class="gh-btn">
          {GITHUB_SVG}
          <span>GitHub</span>
        </a>
      </div>

      <!-- Toolbar -->
      <div class="app-toolbar">
        <button id="tb-upload" class="modern-tb-btn" title="Upload image">
          {UPLOAD_SVG}<span class="tb-label">Upload</span>
        </button>
        <button id="tb-clear" class="modern-tb-btn" title="Clear image">
          {CLEAR_SVG}<span class="tb-label">Clear</span>
        </button>
        <div class="tb-sep"></div>
        <span id="tb-image-count" class="tb-info">No images</span>
      </div>

      <!-- Main row -->
      <div class="app-main-row">

        <!-- Left: image drop + examples -->
        <div class="app-main-left">

          <!-- Drop zone -->
          <div id="gallery-drop-zone">
            <div id="upload-prompt" class="upload-prompt-modern">
              <div id="upload-click-area" class="upload-click-area">
                <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="8" y="14" width="64" height="52" rx="6" fill="none"
                        stroke="#FF1493" stroke-width="2" stroke-dasharray="4 3"/>
                  <polygon points="12,62 30,40 42,50 54,34 68,62"
                           fill="rgba(255,20,147,0.12)" stroke="#FF1493" stroke-width="1.5"/>
                  <circle cx="28" cy="30" r="6"
                          fill="rgba(255,20,147,0.18)" stroke="#FF1493" stroke-width="1.5"/>
                </svg>
                <span class="upload-main-text">Click or drag image here</span>
                <span class="upload-sub-text">Upload the image you want to edit</span>
              </div>
            </div>
            <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />
            <!-- Single image preview -->
            <div id="single-img-container" style="display:none;position:relative;width:100%;height:100%;min-height:440px;cursor:pointer;" title="Click to replace image">
              <img id="single-img-preview" src="" alt="uploaded"
                   style="width:100%;height:100%;min-height:440px;object-fit:contain;display:block;background:#09090b;" />
              <div style="position:absolute;bottom:12px;right:12px;background:rgba(255,20,147,.85);color:#fff;font-size:11px;font-weight:700;padding:4px 12px;border-radius:6px;font-family:'Inter',sans-serif;pointer-events:none;">
                Click to replace
              </div>
            </div>
          </div>

          <!-- Hint -->
          <div class="hint-bar">
            <b>Upload:</b> Click or drag an image &nbsp;&middot;&nbsp;
            <b>Click preview</b> to replace &nbsp;&middot;&nbsp;
            <kbd>Clear</kbd> removes the image
          </div>

          <!-- Quick prompts -->
          <div class="suggestions-section">
            <div class="suggestions-title">Quick Prompts</div>
            <div class="suggestions-wrap">
              <button class="suggestion-chip" onclick="window.__setPrompt('Transform into anime.')">Anime</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Remove shadows and relight using soft lighting.')">Relight</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Rotate the camera 45 degrees to the right.')">Rotate 45°</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Switch the camera to a top-down view.')">Top-down</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Upscale this picture to 4K resolution.')">Upscale 4K</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('flatcolor Desaturate to flat log profile.')">Flat Log</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Light source from the Right Rear.')">Right-Rear Light</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Light source from the Below.')">Below Light</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('dotted illustration.')">Dotted Illus.</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Make the skin details more prominent and natural.')">Edit Skin</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Switch the camera to a bottom-up view.')">Bottom-up</button>
              <button class="suggestion-chip" onclick="window.__setPrompt('Use a subtle golden-hour filter with smooth light diffusion.')">Golden Hour</button>
            </div>
          </div>

          <!-- Examples -->
          <div class="examples-section">
            <div class="examples-title">Quick Examples &mdash; click to load</div>
            <div class="examples-scroll">
              {EXAMPLE_CARDS_HTML}
            </div>
          </div>

        </div><!-- /left -->

        <!-- Right: controls + output -->
        <div class="app-main-right">

          <!-- Prompt -->
          <div class="panel-card">
            <div class="panel-card-title">Edit Instruction</div>
            <div class="panel-card-body">
              <label class="modern-label" for="custom-prompt-input">Prompt</label>
              <textarea id="custom-prompt-input" class="modern-textarea" rows="3"
                        placeholder="e.g., transform into anime, upscale, change lighting&hellip;"></textarea>
            </div>
          </div>

          <!-- LoRA -->
          <div class="lora-selector-card">
            <div class="lora-selector-body">
              <div class="lora-select-label">Editing Style / LoRA</div>
              <select id="custom-lora-select" class="lora-native-select">
                {LORA_OPTIONS_HTML}
              </select>
            </div>
          </div>

          <!-- Run -->
          <div style="padding:14px 18px 6px;">
            <button id="custom-run-btn" class="btn-run">
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C12 2 7 6.5 7 13c0 1.4.3 2.7.8 3.9L5 19.7l1.4 1.4 2.8-2.8c1.2.5 2.5.7 3.8.7s2.6-.2 3.8-.7l2.8 2.8 1.4-1.4-2.8-2.8c.5-1.2.8-2.5.8-3.9 0-6.5-5-11-5-11z"/>
                <circle cx="12" cy="13" r="2"/>
              </svg>
              <span id="run-btn-label">Edit Image</span>
            </button>
          </div>

          <!-- Output -->
          <div class="output-frame" style="flex:1">
            <div class="out-title">
              <span>Output</span>
              <span id="dl-btn-output" class="out-download-btn" title="Download result">
                {DOWNLOAD_SVG} Save
              </span>
            </div>
            <div class="out-body" id="output-image-container">
              <div class="modern-loader" id="output-loader">
                <div class="loader-spinner"></div>
                <div class="loader-text">Processing image&hellip;</div>
                <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
              </div>
              <div class="out-placeholder" id="output-placeholder">Result will appear here</div>
            </div>
          </div>

          <!-- Advanced settings -->
          <div class="settings-group">
            <div class="settings-group-title">Advanced Settings</div>
            <div class="settings-group-body">
              <div class="slider-row">
                <label>Seed</label>
                <input type="range" id="custom-seed" min="0" max="2147483647" step="1" value="0">
                <span class="slider-val" id="custom-seed-val">0</span>
              </div>
              <div class="checkbox-row">
                <input type="checkbox" id="custom-randomize" checked>
                <label for="custom-randomize">Randomize seed</label>
              </div>
              <div class="slider-row">
                <label>Guidance</label>
                <input type="range" id="custom-guidance" min="1" max="10" step="0.1" value="1.0">
                <span class="slider-val" id="custom-guidance-val">1.0</span>
              </div>
              <div class="slider-row">
                <label>Steps</label>
                <input type="range" id="custom-steps" min="1" max="50" step="1" value="4">
                <span class="slider-val" id="custom-steps-val">4</span>
              </div>
            </div>
          </div>

        </div><!-- /right -->
      </div><!-- /main-row -->

      <!-- Footer -->
      <div class="exp-note">
        Experimental Space for
        <a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2509" target="_blank">Qwen-Image-Edit-2509</a>
        &middot; LoRAs loaded lazily on first use
      </div>

      <div class="app-statusbar">
        <div class="sb-section" id="sb-image-count">No images uploaded</div>
        <div class="sb-section sb-fixed">Ready</div>
      </div>

    </div><!-- /app-shell -->
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=None, js=gallery_js)
    demo.load(fn=None, js=wire_outputs_js)

    run_btn.click(
        fn=infer,
        inputs=[hidden_image_b64, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[result, seed],
        js=r"""(img, p, la, s, rs, gs, st) => {
            const imgEl    = document.getElementById('hidden-image-b64');
            const promptEl = document.getElementById('custom-prompt-input');
            const loraEl   = document.getElementById('custom-lora-select');
            const imgVal   = imgEl    ? (imgEl.querySelector('textarea,input') || {}).value || img : img;
            const pVal     = promptEl ? promptEl.value : p;
            const lVal     = loraEl   ? loraEl.value   : la;
            return [imgVal, pVal, lVal, s, rs, gs, st];
        }""",
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples"],
    )
