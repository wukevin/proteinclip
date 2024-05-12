"""Functions for working with PPI data."""

import argparse
import logging
from itertools import chain
import json
from typing import List, Literal, Tuple

import numpy as np
import h5py
from tqdm.auto import tqdm

from proteinclip import data_utils

PPI_DATA_DIR = data_utils.DATA_DIR / "ppi"


def load_ppi_data(
    split: Literal["train", "valid", "test"],
    seed: int = 6489 + 6489,
) -> List[Tuple[str, str, int]]:
    """Load PPI data as pairs of identifiers with a integer label.

    Positive examples are labeled 1, negative examples are labeled 0."""
    prefix = {
        "train": "Intra1",
        "valid": "Intra0",
        "test": "Intra2",
    }[split]

    neg_fname = PPI_DATA_DIR / f"{prefix}_neg_rr.txt"
    pos_fname = PPI_DATA_DIR / f"{prefix}_pos_rr.txt"

    retval = []
    with open(neg_fname) as source:
        for line in source:
            a, b = line.strip().split()
            retval.append((a, b, 0))
    with open(pos_fname) as source:
        for line in source:
            a, b = line.strip().split()
            retval.append((a, b, 1))

    rng = np.random.default_rng(seed)
    rng.shuffle(retval)
    return retval


def make_ppi_data_splits(
    json_out: str, include_train: bool = False, identifiers_only: bool = True
):
    """Create PPI data splits as a .json file. Useful for training models."""
    train = load_ppi_data("train")
    valid = load_ppi_data("valid")
    test = load_ppi_data("test")

    if identifiers_only:
        # Strips out label and flattens the identifiers
        postprocessor = lambda x: sorted(
            set(chain.from_iterable(set((a, b) for a, b, _ in x)))
        )
        train, valid, test = map(postprocessor, (train, valid, test))
        assert not set(train) & set(valid)
        assert not set(train) & set(test)
        assert not set(valid) & set(test)

    splits = {
        "valid": valid,
        "test": test,
    }
    if include_train:
        splits["train"] = train

    with open(json_out, "w") as sink:
        json.dump(splits, sink, indent=2)


def make_ppi_data_splits_for_dsplit(
    out_fname: str,
    embedding_h5: str,
    split: Literal["train", "valid", "test"],
    max_length: int = 0,
):
    """Create PPI data splits as separated .txt files."""
    d = load_ppi_data(split)

    with h5py.File(embedding_h5, "r") as f:
        embed_keys = list(f.keys())
        embed_id_to_key = {k.split("|")[1]: k for k in embed_keys}
        if max_length:
            embed_id_to_length = {
                k.split("|")[1]: np.array(f[k]).shape[1]
                for k in tqdm(embed_keys, desc="Parsing lengths")
            }
        assert len(embed_keys) == len(embed_id_to_key)

    # Function that maps the identifiers tiven to the identifiers present in the
    # embedding h5 file.
    remap = lambda l: [
        (embed_id_to_key[a], embed_id_to_key[b], int(c)) for a, b, c in l
    ]

    # DSplit expects a tsv of [protein 1] [protein 2] [label]
    n_skipped = 0
    with open(out_fname, "w") as sink:
        for raw_entry, entry in zip(d, remap(d)):
            id1, id2, _ = raw_entry
            if max_length and (
                embed_id_to_length[id1] > max_length
                or embed_id_to_length[id2] > max_length
            ):
                n_skipped += 1
                continue
            sink.write("\t".join([str(s) for s in entry]) + "\n")
    if max_length:
        logging.info(f"Skipped {n_skipped} pairs for exceeding {max_length} aa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="Create data split files for DSPLIT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("split", choices=["train", "valid", "test"])
    parser.add_argument("out_fname", type=str, help="Output .tsv to write.")
    parser.add_argument(
        "-e", "--embed", required=True, type=str, help="Embedding .h5py file"
    )
    parser.add_argument(
        "-l", "--length", default=0, type=int, help="Length cap; 0 to disable."
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    make_ppi_data_splits_for_dsplit(
        args.out_fname, args.embed, args.split, max_length=args.length
    )
