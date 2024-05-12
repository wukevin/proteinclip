import os
import json
import argparse
import logging


import torch
from torch.utils import data
import lightning.pytorch as pl
from lightning.pytorch.loggers.csv_logs import CSVLogger

from proteinclip import data_utils, fasta_utils, swissprot, hparams
from proteinclip import contrastive


def write_split_identifiers(train_ids, valid_ids, test_ids, out_file):
    """Write the data split identifiers to the given output .json."""
    with open(out_file, "w") as sink:
        json.dump(
            {
                "train": train_ids,
                "valid": valid_ids,
                "test": test_ids,
            },
            sink,
            indent=4,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Run to train a protein CLIP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "training_config",
        type=os.path.abspath,
        help="Training configuration json file.",
    )
    parser.add_argument(
        "dat",
        type=os.path.abspath,
        help="Data file to read for function text descriptions.",
    )
    parser.add_argument(
        "protein_embed",
        type=os.path.abspath,
        nargs="+",
        help="Protein embeddings as precomputed .hdf5 files mapping <identifier, sequence> to use.",
    )
    parser.add_argument(
        "-s",
        "--splitfile",
        type=lambda s: os.path.abspath(s) if s else s,
        default="",
        help=".json file specifying data splits; splits are random 90/5/5 otherwise.",
    )
    parser.add_argument(
        "--pertoken",
        action="store_true",
        help="Embeddings are per-token, use aggregation.",
    )
    parser.add_argument(
        "--nhidden",
        type=int,
        default=1,
        help="Number of hidden layers in MLP projection networks.",
    )
    parser.add_argument(
        "-d", "--dim", type=int, default=128, help="Shared embedding dim."
    )
    parser.add_argument(
        "-o", "--out", type=str, default=os.getcwd(), help="Output directory."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="protein_clip", help="Name of run."
    )

    parser.add_argument(
        "-g",
        "--gpt",
        type=str,
        choices=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        default="text-embedding-3-small",
        help="GPT embeddings to use.",
    )
    parser.add_argument(
        "--unitnorm",
        action="store_true",
        help="Unit normalize all inputs before feeding to projection network.",
    )
    parser.add_argument(
        "--gpu", type=int, default=1, help="GPU to use (1-indexed); -1 indicates all."
    )
    return parser


def main():
    """Run training script."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(seed=6489)

    # Read in the training config
    hyperparameters = hparams.read_hparams(args.training_config)
    logging.info(f"Hyperparameters: {hyperparameters}")

    # Load precomputed ESM2 embeddings
    esm_embeddings = data_utils.MultiH5(args.protein_embed)

    # Load in the precomputed GPT text embeddings
    sp_text_embed = swissprot.embed_function_descriptions(args.dat, model=args.gpt)

    # Identify shared keys
    shared_keys = sorted(set(esm_embeddings.keys()).intersection(sp_text_embed.keys()))

    # Create dataset; first item is ESM, second is text
    if args.pertoken:
        dset = data_utils.CLIPDataset2D1D(
            pairs=shared_keys, map1=esm_embeddings, map2=sp_text_embed
        )
    else:
        dset = data_utils.CLIPDataset(
            pairs=shared_keys,
            map1=esm_embeddings,
            map2=sp_text_embed,
            enforce_unit_norm=args.unitnorm,
        )

    # Create data splits
    if not args.splitfile:
        split_indices = data_utils.random_split(len(dset), [0.9, 0.05, 0.05])
        logging.info(f"Randomized split sizes: {[len(x) for x in split_indices]}")
    else:
        assert os.path.isfile(args.splitfile)
        with open(args.splitfile, "r") as source:
            splits = json.load(source)
        assert "valid" in splits and "test" in splits
        if "train" not in splits:
            logging.warning(
                "No 'train' in splits; using all non-valid/test pairs to train."
            )
            splits["train"] = [
                p
                for p in dset.pairs
                if p not in splits["valid"] and p not in splits["test"]
            ]
        logging.info(
            f"Loaded split IDs: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}"
        )
        # Pre-cache the mapping for identifier -> index
        id2index = {p: i for i, p in enumerate(dset.pairs)}
        split_indices = [
            [id2index[p] for p in splits[s] if p in id2index]
            for s in ("train", "valid", "test")
        ]
    dset_splits = [data.Subset(dset, idx) for idx in split_indices]

    # Create data loaders
    train_dl, valid_dl, _test_dl = [
        data.DataLoader(
            ds,
            batch_size=hyperparameters.batch_size,
            shuffle=(i == 0),
            # drop_last=(i == 0),
            num_workers=8,
            pin_memory=True,
        )
        for i, ds in enumerate(dset_splits)
    ]

    # Definte network
    input_dim_1 = next(iter(train_dl))["x_1"].shape[-1]
    input_dim_2 = next(iter(train_dl))["x_2"].shape[-1]
    model_class = (
        contrastive.ContrastiveEmbeddingWithPreprocessor
        if args.pertoken
        else contrastive.ContrastiveEmbedding
    )
    net = model_class(
        input_dim_1=input_dim_1,
        input_dim_2=input_dim_2,
        shared_dim=args.dim,
        num_hidden=args.nhidden,
        lr=hyperparameters.learning_rate,
    )

    # Define logger, write configuration files and data splits
    logger = CSVLogger(save_dir=args.out, name=args.name)
    logger.log_hyperparams(hyperparameters.as_dict())
    write_split_identifiers(
        train_ids=[dset.pairs[i] for i in split_indices[0]],
        valid_ids=[dset.pairs[i] for i in split_indices[1]],
        test_ids=[dset.pairs[i] for i in split_indices[2]],
        out_file=os.path.join(logger.log_dir, "data_splits.json"),
    )
    net.write_config_json(os.path.join(logger.log_dir, "model_config.json"))
    with open(os.path.join(logger.log_dir, "training_config.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)

    # Train
    trainer = pl.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="cuda",
        devices=args.gpu,
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    # Export model as ONNX files
    contrastive.model_to_onnx(
        net.project_1,
        os.path.join(logger.log_dir, "project_1.onnx"),
        input_shape=(input_dim_1,),
    )
    contrastive.model_to_onnx(
        net.project_2,
        os.path.join(logger.log_dir, "project_2.onnx"),
        input_shape=(input_dim_2,),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
