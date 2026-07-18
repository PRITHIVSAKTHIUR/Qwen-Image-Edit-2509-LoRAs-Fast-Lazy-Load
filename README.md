# **[Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2509-LoRAs-Fast)**

Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load is an experimental, high-performance image editing and style-transfer sandbox built on top of the `Qwen/Qwen-Image-Edit-2509` base model. Powered by a deeply optimized transformer backbone (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4`) and accelerated via Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`), this application delivers ultra-fast, 4-step image refinement directly in pixel space.

The application features an adaptive **Lazy Loading architecture** for LoRA adapters. Weights for its 11+ specialized multi-angle, relighting, and illustration models are pulled and hot-fused into the inference state dynamically only upon request. The system is wrapped in a highly responsive, custom Iris-themed UI containing client-side JavaScript drag-and-drop mechanics, immediate example loading, and automated image parameter snapping.

### **Key Features**

* **Lazy-Loaded LoRA Hub:** On-demand downloading and weight-fusing for 11+ pre-configured specialized LoRA adapters (including *Photo-to-Anime*, *Multiple-Angles*, *Light-Restoration*, *Relight*, *Multi-Angle-Lighting*, *Edit-Skin*, *Next-Scene*, *Flat-Log*, *Upscale-Image*, *Upscale2K*, and *Dotted-Illustration*).
* **Flash Attention 3 (FA3) Acceleration:** Wire-fused with a custom `QwenDoubleStreamAttnProcessorFA3` handler to drastically compress VRAM allocation and boost step-by-step cross-attention calculations.
* **Smart Dimension Snapping:** Automatically scales uploaded reference images so that the longer side stays under 1024px while snapping both width and height to strict multiples of 8, preventing tensor geometry faults.
* **Polished In-App Shell UI:** Replaces generic block items with a premium web console featuring deep slate and iris gradients, responsive action menus, custom suggestion chips, and zero-flicker launch button animations.
* **Transient VRAM Governance:** Tailored for seamless integration with `spaces.GPU` context streams, executing full python-side trash collections (`gc.collect()`) and cache purges before and after every diffusion run.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 10.jpeg
│   ├── 11.jpg
│   ├── 12.jpg
│   ├── 13.jpg
│   ├── 14.jpg
│   ├── 2.jpeg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   ├── 9.jpg
│   ├── DI.jpg
│   └── ELS.jpg
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To initialize the Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load space locally, verify your host is configured with the specialized environment and compiled dependencies detailed below. A dedicated CUDA-compatible GPU is required.

* **Python Version:** Minimum Python **3.12** is needed, though Python **3.14** is recommended for peak optimization.
* **PyTorch Version:** `torch==2.11.0` or above is required for cross-attention architecture compatibility.
* **CUDA Hardware Layer:** CUDA **13.0** is recommended to mirror the live Hugging Face production environment.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, which ensures rapid virtual environment synchronization and reproducible dependency execution loops.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh`
* **Windows:** `powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git
cd Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

#### **Standard PIP Installation**

**1. Upgrade Package Manager**
Update your system installer prior to fetching modern pre-compiled wheel distributions:

```bash
pip install pip>=26.1.2

```

**2. Core Dependency Pull**
Install the primary deep learning stack, core transformer builds, and layout engine libraries from your local `requirements.txt` file:

```bash
pip install -r requirements.txt

```

#### **Core Requirements List (`requirements.txt`)**

```text
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
tokenizers==0.22.2
sentencepiece==0.2.2
safetensors==0.8.0
gradio==6.20.0
gradio-client==2.5.0
hf-gradio==0.4.1
spaces==0.51.0
fastapi==0.139.1
starlette==1.3.1
uvicorn==0.51.0
pydantic==2.12.5
pydantic-core==2.41.5
pydantic-settings==2.14.2
typing-inspection==0.4.2
python-multipart==0.0.32
orjson==3.11.9
httpx-sse==0.4.3
websockets==16.1
mcp==1.28.1
platformdirs==4.10.0
psutil==7.2.2
regex==2026.7.10
pillow==12.3.0
av==18.0.0
pydub==0.25.1
authlib==1.7.2
cryptography==49.0.0
pyOpenSSL==26.3.0
cffi==2.1.0
pycparser==3.0
email-validator==2.3.0
dnspython==2.8.0
python-dotenv==1.2.2
itsdangerous==2.2.0
pyjwt==2.13.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
referencing==0.37.0
rpds-py==2026.6.3
annotated-types==0.7.0
semantic-version==2.10.0
tomlkit==0.14.0
importlib_metadata==9.0.0
zipp==4.1.0
pytz==2026.2
safehttpx==0.1.7
brotli==1.2.0
groovy==0.1.2
id==1.6.1
joserfc==1.7.3
kernels==0.16.0
kernels-data==0.16.0
rfc3161-client==1.0.7
rfc8785==0.1.4
securesystemslib==1.4.0
sigstore==4.4.0
sigstore-models==0.0.6
sigstore-rekor-types==0.0.18
sse-starlette==3.4.5
tuf==7.0.0

# scientific computing
numpy==2.4.6
```

### **Usage**

Once the web server deployment initiates, load the app locally by pointing your browser to the local network link (typically `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`).

1. **Upload Asset:** Drop your source photo or asset directly into the dashed Iris uploader area. Clicking on an active image preview allows you to instantly replace it with a new file.
2. **Select Style/LoRA:** Use the **Editing Style / LoRA** dropdown selector to choose your target adapter. The model will automatically trigger a lazy-load download of the chosen weights on its first use.
3. **Refine Prompt:** Enter your specific edit instructions (e.g., *"Transform into anime"* or *"Rotate the camera 45 degrees to the left"*), or click on any of the **Quick Prompts** chips to instantly fill the input box.
4. **Execute:** Click **Edit Image**. The loader overlay will animate, and once the 4-step diffusion pass completes, the final image will render with an option to download the output as a clean `.png` file.

### **Links and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load/blob/main/LICENSE)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2509-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2509-LoRAs-Fast)
