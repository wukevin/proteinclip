import os
import argparse
import json
from glob import glob
import logging
from pathlib import Path
from typing import Any, Dict, Literal, List, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from tqdm.auto import tqdm

from proteinclip import contrastive
from proteinclip import data_utils
from proteinclip import hparams

from proteinclip.ppi_data import load_ppi_data

PPI_DATA_DIR = data_utils.DATA_DIR / "ppi"


def infer_esm_size_from_path(path: str) -> int:
    bname = os.path.basename(path)
    if "33layer" in bname and "30layer" not in bname:
        return 33
    elif "30layer" in bname and "33layer" not in bname:
        return 30
    raise ValueError(f"Could not infer ESM size from path {path}")


def load_model_config(model_dir: Path | str) -> Dict[str, Any]:
    """Load model and training configuration."""
    model_config_path = Path(model_dir) / "model_config.json"
    if model_config_path.exists():
        with open(model_config_path) as source:
            model_config = json.load(source)

    training_config_path = Path(model_dir) / "training_config.json"
    with open(training_config_path) as source:
        training_config = json.load(source)
    return model_config, training_config


def build_parser():
    """Build a CLI argument parser."""
    parser = argparse.ArgumentParser(
        usage="Train a protein-protein interaction network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "training_config", type=str, help="Training configuration json file."
    )
    parser.add_argument(
        "-c",
        "--clip",
        type=str,
        required=True,
        help="Path to a directory that contains trained CLIP model",
    )
    parser.add_argument(
        "--clipnum",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Whether CLIP model uses projection 1 or 2 for protein embeddings; 0 to disable projection.",
    )
    parser.add_argument(
        "-o", "--out", type=str, default=os.getcwd(), help="Output directory."
    )
    parser.add_argument("-n", "--name", type=str, default="ppi", help="Name of run.")
    return parser


def main():
    """Run training script."""
    parser = build_parser()
    args = parser.parse_args()

    # Load in hyperparameters
    hyperparameters = hparams.read_hparams(args.training_config)
    logging.info(f"Hyperparameters: {hyperparameters}")

    # Load in the model and training configuration
    clip_model_config, clip_training_config = load_model_config(args.clip)

    contrastive_net = contrastive.ContrastiveEmbedding.load_from_checkpoint(
        glob(os.path.join(args.clip, "checkpoints/epoch=*-step=*.ckpt")).pop(),
        **clip_model_config,
    ).eval()

    # Load in the pairs
    train_pairs, valid_pairs, test_pairs = (
        load_ppi_data(split) for split in ("train", "valid", "test")
    )
    all_keys = set()
    for p in train_pairs + valid_pairs + test_pairs:
        all_keys.add(p[0])
        all_keys.add(p[1])

    # Load in the embeddings that we are using based on ESM size; filter to only
    # keep embeddings that are present in interaction pairs
    base_path = Path("/home/wukevin/projects/gpt-protein-results")
    full_embed_map = data_utils.MultiH5(
        [
            base_path / p
            for p in (
                clip_training_config.get("esm")
                or clip_training_config.get("protein_embed")
            )
        ]
    )
    embed_map = {
        k: v
        for k, v in tqdm(full_embed_map.items(), desc="Loading relevant embeddings")
        if k in all_keys
    }
    logging.info(f"Loaded {len(embed_map)} relevant embeddings")

    # Handle unit norming
    if clip_training_config.get("unitnorm", False):
        logging.info("Unit normalizing embeddings")
        embed_map = {k: v / np.linalg.norm(v) for k, v in embed_map.items()}

    # Project embeddings
    if args.clipnum:
        dev = next(iter(contrastive_net.parameters())).device
        embed_func = (
            contrastive_net.project_1
            if args.clipnum == 1
            else contrastive_net.project_2
        )
        logging.info(f"Reprojecting embeddings with {embed_func}")
        with torch.no_grad():
            embed_map = {
                k: embed_func(torch.from_numpy(np.array(v)).to(dev)).cpu().numpy()
                for k, v in tqdm(embed_map.items(), desc="Reprojection")
            }
    else:
        logging.warning("Not reprojecting embeddings; using raw embeddings!")

    train_dset, valid_dset, test_dset = [
        data_utils.LabeledPairDataset(p, embed_map)
        for p in (train_pairs, valid_pairs, test_pairs)
    ]
    logging.info(
        f"Train/valid/test sizes: {len(train_dset)}/{len(valid_dset)}/{len(test_dset)}"
    )
    train_dl, valid_dl, test_dl = [
        DataLoader(
            d, batch_size=hyperparameters.batch_size, shuffle=(i == 0), num_workers=16
        )
        for i, d in enumerate([train_dset, valid_dset, test_dset])
    ]

    # Define network and training infrastructure
    pl.seed_everything(seed=6489)

    net = contrastive.MLPForClassification(
        init_dim=next(iter(train_dl))[0].shape[-1],
        dims=[128, 1],
        lr=hyperparameters.learning_rate,
        unit_norm=False,
    )

    logger = CSVLogger(save_dir=args.out, name=args.name)
    logger.log_hyperparams(hyperparameters.as_dict())
    # net.write_config_json(os.path.join(logger.log_dir, "model_config.json"))
    with open(os.path.join(logger.log_dir, "training_config.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)

    ckpt_callback = ModelCheckpoint(
        dirpath=None,
        filename="{epoch}-{val_auprc:.4f}",
        monitor="val_auprc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=True,
    )
    trainer = pl.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="cuda",
        enable_progress_bar=True,
        logger=logger,
        callbacks=[ckpt_callback],
        log_every_n_steps=1,
        deterministic=True,
    )
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    # Read in metrics and report best validation auprc
    metrics = pd.read_csv(os.path.join(logger.log_dir, "metrics.csv"))
    logging.info(f"Best validation AUPRC: {metrics['val_auprc'].max()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
