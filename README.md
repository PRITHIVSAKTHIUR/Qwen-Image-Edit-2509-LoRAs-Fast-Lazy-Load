# Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load

> A Gradio-based demonstration for the Qwen/Qwen-Image-Edit-2509 model, featuring lazy-loaded LoRA adapters for fast, specialized image edits like photo-to-anime conversion, angle changes, lighting restoration, skin editing, and upscaling. Supports single-image inputs with descriptive prompts; adapters load on-demand to optimize memory. Enhanced with Flash Attention 3 for efficient inference.

## Features

- **Lazy LoRA Loading**: 8 specialized adapters (e.g., Photo-to-Anime, Relight) download and activate only when selected, reducing initial load time.
- **Single-Image Editing**: Upload one image and apply edits via prompts (e.g., "Transform into anime").
- **Rapid Inference**: 4-step default generations with bfloat16 and optional Flash Attention 3; auto-resizes outputs to match aspect (multiples of 8).
- **Advanced Controls**: Hidden accordion for seed randomization, guidance scale (1-10), and steps (1-50).
- **Custom Theme**: OrangeRedTheme with gradients and responsive CSS for a clean UI.
- **Examples**: 15 pre-loaded inputs for quick testing (e.g., multi-angle views, next-scene transitions).
- **Queueing Support**: Up to 30 concurrent jobs with 300s cache for smooth usage.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU).
- Stable internet for initial model/LoRA downloads.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git
   cd Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/diffusers.git
   git+https://github.com/huggingface/peft.git
   huggingface_hub
   sentencepiece
   transformers
   torchvision
   supervision
   kernels
   gradio #gradio@6.3.0
   spaces
   torch
   numpy
   av
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Upload Image**: Select a single image (PIL format, height up to 290px preview).

2. **Select Adapter**: Dropdown for styles (default: "Photo-to-Anime").

3. **Enter Prompt**: Describe the edit (e.g., "Rotate the camera 45 degrees to the left").

4. **Configure (Optional)**: Expand "Advanced Settings" for seed, guidance, steps.

5. **Edit Image**: Click "Edit Image"; outputs the transformed image with seed displayed.

### Supported Adapters

| Adapter              | Use Case                          |
|----------------------|-----------------------------------|
| Photo-to-Anime      | Style transfer to anime aesthetics |
| Multiple-Angles     | Camera rotation/view changes     |
| Light-Restoration   | Shadow removal and relighting    |
| Relight             | Lighting adjustments             |
| Multi-Angle-Lighting| Dynamic multi-view lighting      |
| Edit-Skin           | Skin detail enhancement          |
| Next-Scene          | Sequential scene transitions     |
| Upscale-Image       | Resolution upscaling             |

## Examples

| Input Image    | Prompt Example                                      | Adapter              |
|----------------|-----------------------------------------------------|----------------------|
| examples/1.jpg | "Transform into anime."                            | Photo-to-Anime      |
| examples/5.jpg | "Remove shadows and relight using soft lighting."  | Light-Restoration   |
| examples/4.jpg | "Use a subtle golden-hour filter."                 | Relight             |
| examples/2.jpeg| "Rotate the camera 45 degrees to the left."        | Multiple-Angles     |
| examples/7.jpg | "Light source from the Right Rear."                | Multi-Angle-Lighting|
| examples/10.jpeg| "Upscale the image."                               | Upscale-Image       |
| examples/7.jpg | "Light source from the Below."                     | Multi-Angle-Lighting|
| examples/2.jpeg| "Switch to a top-down right corner view."          | Multiple-Angles     |
| examples/9.jpg | "Camera moves forward as sunlight breaks through." | Next-Scene          |
| examples/8.jpg | "Make skin details more prominent and natural."    | Edit-Skin           |
| examples/6.jpg | "Switch to a bottom-up view."                      | Multiple-Angles     |
| examples/6.jpg | "Rotate the camera 180 degrees upside down."       | Multiple-Angles     |
| examples/4.jpg | "Rotate the camera 45 degrees to the right."       | Multiple-Angles     |
| examples/4.jpg | "Switch to a top-down view."                       | Multiple-Angles     |
| examples/4.jpg | "Switch to a wide-angle lens."                     | Multiple-Angles     |

## Troubleshooting

- **Adapter Loading Errors**: First selection downloads LoRA; check internet/repo. Console logs status.
- **OOM on GPU**: Reduce steps/resolution; clear cache with `torch.cuda.empty_cache()`. FA3 fallback if incompatible.
- **Dimension Issues**: Auto-resizes to 1024 max edge (aspect preserved); multiples of 8 enforced.
- **No Output**: Ensure image uploaded; prompts must be descriptive.
- **Queue Full**: Increase `max_size` in `demo.queue()`; 300s cache for edits.
- **Gradio Rendering**: Set `ssr_mode=True` if gradients fail; CSS for container width.

## Contributing

Contributions encouraged! Fork the repo, add adapters to `ADAPTER_SPECS`, or enhance prompts, and submit PRs with tests. Focus areas:
- More LoRA integrations.
- Batch/single-step modes.
- Negative prompt customization.

Repository: [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
