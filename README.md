# SMART-Edit: Spectral Mutual Attention Refinement and Fusion for Precise Training-Free Image Editing

Official implementation of **SMART-Edit: Spectral Mutual Attention Refinement and Fusion for Precise Training-Free Image Editing**, built on [🤗 Diffusers](https://github.com/huggingface/diffusers).

SMART-Edit enables high-precision training-free image editing via spectral mutual attention refinement and fusion.

---

## ✨ Features

- **Training-Free Editing**: No fine-tuning required, edit images directly with text prompts.
- **Spectral Mutual Attention**: Refines attention maps in the spectral domain to preserve source structure and align with target prompts.
- **Integrates SOTA Methods**: Built upon and unifies multiple existing state-of-the-art RF-based editing methods:
  - [Vanilla](https://arxiv.org/abs/2410.10792) (project baseline, RF-Inversion vanilla variant)
  - [RF-Inversion](https://arxiv.org/abs/2410.10792) (ICLR 2024)
  - [RF-Solver](https://arxiv.org/abs/2411.04746) (ICML 2025)
  - [FireFlow](https://arxiv.org/abs/2412.07517) (ICML 2025)
  - [FTEdit](https://arxiv.org/abs/2411.15843) (CVPR 2025)
  - [FlowEdit](https://arxiv.org/abs/2412.08629) (ICCV 2025)
  - [DNAEdit](https://arxiv.org/abs/2506.01430) (NeurIPS 2025)
  - [FSIEdit](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c0d7b7a75dc49664bfc6ac9cb935792d-Abstract-Conference.html) (NeurIPS 2025)
  - [FIAEdit](https://arxiv.org/pdf/2511.12151) (AAAI 2026)
- **Supports popular models**: Compatible with SD3.5 and FLUX.1 pre-trained diffusion models.
- **Unified repository**: Integrates multiple SOTA editing methods into a consistent framework.
- **Supports Multiple Scenarios**: Object replacement, attribute modification, sequential editing.

---

## ⚙️ Installation

```bash
uv sync
source .venv/bin/activate
# or
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
import diffusers
import torch
from pipeline import SMART_EditDNAEditFluxPipeline

diffusers.utils.logging.set_verbosity_error()
pipe = SMART_EditDNAEditFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.add_processor(after_layer=0, before_layer=37, filter_name="single_transformer_blocks")

image = pipe(
    source_img="assets/sources/cat.png",
    source_prompt="portrait of a cat",
    target_prompt="portrait of a tiger",
    source_guidance_scale=1.0,
    target_guidance_scale=2.5,
    start_timestep=4,
    inject_step=4,
    mask_prompt="cat",
    num_inference_steps=28,
).images[0]
image.save("output.jpg")
```

---

## 📦 Dataset

The repository supports evaluation on two widely-used image-editing benchmarks:

- **PIE-Bench** ([PnPInversion](https://github.com/cure-lab/PnPInversion))
  A collection of ~700 editing samples covering background, style, object, and attribute changes. Used as the default benchmark in `config/exp.yaml` (`datasets.pie_bench`). Update `data_root` to your local PIE-Bench_v1 path before running.

- **EditEval** ([Awesome-Diffusion-Model-Based-Image-Editing-Methods](https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods))
  A curated benchmark with diverse prompts and reference images, designed to evaluate generalization of diffusion-based editing methods.

Download the datasets from the official repositories above, then point the corresponding `data_root` in your config to the local path.

---

## 💡 Core Idea

1. **Spectral Mutual Attention Refinement**: Better align attention maps with target prompts while preserving source structure.
2. **Attention Fusion**: Fuses refined source/target attention to achieve precise editing with minimal artifacts.

Core code:
- Attention processing: [`processor/`](processor/)
- Editing pipeline: [`pipeline/`](pipeline/)

---

## 📝 Citation

Citation for SMART-Edit will be added once the paper is published. If you use this code, please cite the original works of the integrated methods listed above.

---

## 🙏 Acknowledgements

Thanks to the authors of the integrated methods for open-sourcing their work. This project is based on the [RF-Image-Editing](https://github.com/Justin900429/RF-Image-Editing) framework.
