"""
Script to take a pre-trained CLIP model and extract the different projection
networks from it.

This is useful for repurposing the projections in other applications/networks
that may not have the same set of dependencies as the original CLIP model.
"""

import os
from glob import glob
import argparse

from proteinclip import contrastive

from train_ppi import load_model_config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract projection networks from a pre-trained CLIP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "clip",
        type=str,
        help="Path to CLIP model to extract projections. Should be a folder contining metrics.csv and checkpoints/ folder.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Path to directory to save extracted projections. Defaults to subfolder 'projections' under CLIP dir.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    clip_model_config, _clip_training_config = load_model_config(args.clip)

    contrastive_net = (
        contrastive.ContrastiveEmbedding.load_from_checkpoint(
            glob(os.path.join(args.clip, "checkpoints/epoch=*-step=*.ckpt")).pop(),
            **clip_model_config,
        )
        .eval()
        .cpu()
    )

    if not args.output_dir:
        args.output_dir = os.path.join(args.clip, "projections")
    os.makedirs(args.output_dir, exist_ok=True)

    contrastive.model_to_onnx(
        contrastive_net.project_1, os.path.join(args.output_dir, "project_1.onnx")
    )
    contrastive.model_to_onnx(
        contrastive_net.project_2, os.path.join(args.output_dir, "project_2.onnx")
    )


if __name__ == "__main__":
    main()
