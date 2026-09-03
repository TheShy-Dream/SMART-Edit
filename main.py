import argparse
import torch
from pathlib import Path
from torch.utils import data
import os
import sys
from pipeline import *

import diffusers
# 初始化Flux模型和pipeline

# Method -> [Flux class name, SD35 class name]. Resolved at runtime via
# globals() so methods whose SD35 implementation is missing (some methods
# only ship with the Flux variant) become unavailable instead of crashing
# import time.
_METHOD_PAIR_NAMES = {
    "rf_solver":            ("RFSolverEditFluxPipeline",            "RFSolverEditSD35Pipeline"),
    "rf_inversion":         ("RFInversionEditFluxPipeline",         "RFInversionEditSD35Pipeline"),
    "fireflow_edit":        ("FireFlowEditFluxPipeline",           "FireFlowEditSD35Pipeline"),
    "flow_edit":            ("FlowEditFluxPipeline",               "FlowEditSD35Pipeline"),
    "rf_flow_vanilla":      ("RFVanillaFluxPipeline",              "RFVanillaSD35Pipeline"),
    "ft_edit":              ("FTEditFluxPipeline",                 "FTEditSD35Pipeline"),
    "dna_edit":             ("DNAEditFluxPipeline",                "DNAEditSD35Pipeline"),
    "fia_edit":             ("RFFIAEditFluxPipeline",              "RFFIAEditSD35Pipeline"),
    "fsi_edit":             ("RFFSIEditFluxPipeline",              "RFFSIEditSD35Pipeline"),
    "smart_edit":           ("SMART_EditEditFluxPipeline",         "SMART_EditEditSD35Pipeline"),
    "smart_edit_dna":       ("SMART_EditDNAEditFluxPipeline",      "SMART_EditDNAEditSD35Pipeline"),
    "smart_edit_fireflow":  ("SMART_EditFireFlowEditFluxPipeline", "SMART_EditFireFlowEditSD35Pipeline"),
    "smart_edit_fia":       ("SMART_EditFIAEditFluxPipeline",      "SMART_EditFIAEditSD35Pipeline"),
    "smart_edit_fsi":       ("SMART_EditFSIEditFluxPipeline",      "SMART_EditFSIEditSD35Pipeline"),
    "smart_edit_ft":        ("SMART_EditFTEditFluxPipeline",       "SMART_EditFTEditSD35Pipeline"),
    "smart_edit_rf_inversion": ("SMART_EditRFInversionEditFluxPipeline", "SMART_EditRFInversionEditSD35Pipeline"),
    "smart_edit_rf_solver":   ("SMART_EditRFSolverEditFluxPipeline",    "SMART_EditRFSolverEditSD35Pipeline"),
    "smart_edit_vanilla":     ("SMART_EditVanillaFluxPipeline",         "SMART_EditVanillaSD35Pipeline"),
}

_FLUX_BY_METHOD = {
    method: globals()[f_name]
    for method, (f_name, _s_name) in _METHOD_PAIR_NAMES.items()
    if f_name in globals()
}
_SD35_BY_METHOD = {
    method: globals()[s_name]
    for method, (_f_name, s_name) in _METHOD_PAIR_NAMES.items()
    if s_name in globals()
}
AVAILABLE_METHODS_FLUX = sorted(_FLUX_BY_METHOD.keys())
AVAILABLE_METHODS_SD35 = sorted(_SD35_BY_METHOD.keys())


def get_pipeline_class(method: str, backbone: str):
    """Return the editing pipeline class for (method, backbone).

    Raises KeyError if the requested backbone has no implementation for the
    method, so callers can skip the combination gracefully.
    """
    if backbone == "flux":
        if method not in _FLUX_BY_METHOD:
            raise KeyError(method)
        return _FLUX_BY_METHOD[method]
    if backbone == "sd35":
        if method not in _SD35_BY_METHOD:
            raise KeyError(method)
        return _SD35_BY_METHOD[method]
    raise ValueError(f"Unsupported backbone: {backbone}")

# 添加项目路径到sys.path
parent_dir = os.path.abspath(os.getcwd())

def single_sample_mode(args):
    """Single sample editing mode - supports one image generation"""
    diffusers.utils.logging.set_verbosity_error()

    print(f"Using backbone: {args.backbone}")
    print(f"Using method: {args.method}")
    print(f"Source image: {args.source_img}")
    print(f"Source prompt: {args.source_prompt}")
    print(f"Target prompt: {args.target_prompt}")
    if args.method.startswith("smart_edit"):
        print(f"Mask prompt: {args.mask_prompt}")

    # Load config based on backbone
    import yaml
    if args.backbone == 'flux':
        model_config = yaml.safe_load(open("config/flux_exp.yaml"))
    elif args.backbone == 'sd35':
        model_config = yaml.safe_load(open("config/SD35_exp.yaml"))
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    try:
        pipeline_class = get_pipeline_class(args.method, args.backbone)
    except KeyError:
        raise ValueError(
            f"Method {args.method!r} has no {args.backbone} implementation in this repo. "
            f"Flux: {AVAILABLE_METHODS_FLUX}\n"
            f"SD35: {AVAILABLE_METHODS_SD35}"
        )

    method_config = model_config['methods'][args.method]

    # 方法级 model_path 可覆盖全局 model.path（用于 FlowEdit/SMART-Edit flow-style 用 SD3 等场景）
    model_path = method_config.get('model_path', model_config['model']['path'])

    # Load pipeline from HuggingFace (standard, no quantization)
    pipe = pipeline_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    device = model_config.get('general', {}).get('device', 'cuda')
    pipe.to(device)

    # Add processor if needed
    if method_config.get('parameters').get('add_processor'):
        add_processor_config = method_config.get('parameters')['add_processor']
        pipe.add_processor(**add_processor_config)

    # Add controller if needed
    if method_config.get('parameters').get('controller'):
        controller_config = method_config.get('parameters')['controller']
        controller_type = controller_config.get('type')
        if controller_type == 'FluxAttentionReplace':
            from processor.ft_editing_attn_processor import FluxAttentionReplace, P2PFlux_JointAttnProcessor2_0
            controller = FluxAttentionReplace(
                prompts=controller_config.get('prompts',[""]),
                num_steps=controller_config.get('num_steps', 25),
                attn_ratio=controller_config.get('attn_ratio', 0.15),
                num_att_layers=controller_config.get('num_att_layers', 38),
            )
            pipe.add_processor(
                after_layer=0,
                before_layer=37,
                filter_name="single_transformer_blocks",
                target_processor=P2PFlux_JointAttnProcessor2_0,
                controller=controller
            )
        elif controller_type == 'SD35AttentionReplace':
            from processor.ft_editing_attn_processor import SD3AttentionReplace, P2P35_JointAttnProcessor2_0
            prompts = controller_config.get('prompts', ["", "", "", ""])
            controller = SD3AttentionReplace(
                prompts=prompts,
                num_steps=controller_config.get('num_steps', 25),
                attn_ratio=controller_config.get('attn_ratio', 0.15),
                num_att_layers=controller_config.get('num_att_layers', 18),
            )
            pipe.add_processor(
                after_layer=0,
                before_layer=17,
                filter_name="transformer_blocks",
                target_processor=P2P35_JointAttnProcessor2_0,
                controller=controller
            )
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

    # Get default parameters
    width = model_config.get('general', {}).get('width', 512)
    height = model_config.get('general', {}).get('height', 512)
    negative_prompt = model_config.get('general', {}).get('negative_prompt',"") if args.backbone == "sd35" else None

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Run inference
    print(f"Generating edited image...")

    if args.backbone == "sd35":
        if args.method.startswith("smart_edit"):
            # SMART-Edit on SD35 takes source_prompt / target_prompt / negative_prompt
            # positionally; mask_prompt is converted to token_indices by the pipeline.
            result = pipe(
                args.source_img,
                args.source_prompt,
                args.target_prompt,
                negative_prompt,
                mask_prompt=args.mask_prompt,
                width=width,
                height=height,
                **{k: v for k, v in method_config['parameters'].items() if k not in ["add_processor", "controller"]}
            )
        else:
            result = pipe(
                args.source_img,
                args.source_prompt,
                args.target_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                **{k: v for k, v in method_config['parameters'].items() if k not in ["add_processor", "controller"]}
            )
    else:  # flux
        if args.method.startswith("smart_edit"):
            # SMART-Edit on Flux: pass mask_prompt as a kwarg so the pipeline
            # can resolve token_indices from it.
            result = pipe(
                args.source_img,
                args.source_prompt,
                args.target_prompt,
                width=width,
                height=height,
                mask_prompt=args.mask_prompt,
                **{k: v for k, v in method_config['parameters'].items() if k not in ["add_processor", "controller"]}
            )
        else:
            result = pipe(
                args.source_img,
                args.source_prompt,
                args.target_prompt,
                width=width,
                height=height,
                **{k: v for k, v in method_config['parameters'].items() if k not in ["add_processor", "controller"]}
            )

    edited_image = result.images[0]
    edited_image.save(args.save_path)
    print(f"Saved result to: {args.save_path}")

def main():
    parser = argparse.ArgumentParser(description="SMART-Edit - Single sample image editing")

    # Single sample mode arguments
    parser.add_argument('--backbone', type=str, choices=['flux', 'sd35'], default='flux',
                      help='Backbone model (flux or sd35)')
    parser.add_argument('--method', type=str, default='smart_edit_fsi',
                      help='Editing method')
    parser.add_argument('--source_img', type=str, default=str(Path(parent_dir) / "assets/sources/coffee.jpg"),
                      help='Path to source image')
    parser.add_argument('--save_path', type=str, default=str(Path(parent_dir) / "assets/results/coffee_flux_smart_edit_fsi.jpg"),
                      help='Path to save edited image')
    parser.add_argument('--mask_prompt', type=str, default='tulip',
                      help='Mask prompt for SMART-Edit methods')
    parser.add_argument('--source_prompt', type=str, default='a cup of coffee with a drawing of a tulip put on the wooden table.',
                      help='Source prompt')
    parser.add_argument('--target_prompt', type=str, default='a cup of coffee with a drawing of a lion put on the wooden table.',
                      help='Target prompt')

    args = parser.parse_args()
    single_sample_mode(args)

if __name__ == "__main__":
    main()
