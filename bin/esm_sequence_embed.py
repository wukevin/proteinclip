"""
Script to embed sequences using ESM model.

Note that this is NOT the same as embedding a mutation.
"""

# Commands for 6-layer model
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa data/swissprot/esm_embed_6/esm_6layer_split0.hdf5 --model 6 --maxlen 5800 --gpu 0
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa data/swissprot/esm_embed_6/esm_6layer_split1.hdf5 --model 6 --maxlen 5800 --gpu 1
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa data/swissprot/esm_embed_6/esm_6layer_split2.hdf5 --model 6 --maxlen 5800 --gpu 2
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa data/swissprot/esm_embed_6/esm_6layer_split3.hdf5 --model 6 --maxlen 5800 --gpu 3

# Commands for 12-layer model
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa data/swissprot/esm_embed_12/esm_12layer_split0.hdf5 --model 12 --maxlen 5800 --gpu 0
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa data/swissprot/esm_embed_12/esm_12layer_split1.hdf5 --model 12 --maxlen 5800 --gpu 1
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa data/swissprot/esm_embed_12/esm_12layer_split2.hdf5 --model 12 --maxlen 5800 --gpu 2
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa data/swissprot/esm_embed_12/esm_12layer_split3.hdf5 --model 12 --maxlen 5800 --gpu 3

# Commands for 30-layer model
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa data/swissprot/esm_embed_30/esm_30layer_split0.hdf5 --model 30 --maxlen 5800 --gpu 0
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa data/swissprot/esm_embed_30/esm_30layer_split1.hdf5 --model 30 --maxlen 5800 --gpu 1
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa data/swissprot/esm_embed_30/esm_30layer_split2.hdf5 --model 30 --maxlen 5800 --gpu 2
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa data/swissprot/esm_embed_30/esm_30layer_split3.hdf5 --model 30 --maxlen 5800 --gpu 3

# Commands for 33-layer model
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa data/swissprot/esm_embed_33/esm_33layer_split0.hdf5 --model 33 --maxlen 5800 --gpu 0
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa data/swissprot/esm_embed_33/esm_33layer_split1.hdf5 --model 33 --maxlen 5800 --gpu 1
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa data/swissprot/esm_embed_33/esm_33layer_split2.hdf5 --model 33 --maxlen 5800 --gpu 2
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa data/swissprot/esm_embed_33/esm_33layer_split3.hdf5 --model 33 --maxlen 5800 --gpu 3

# Commands for 33-layer model, per-token
# python ~/projects/gpt-protein/bin/esm_sequence_embed.py ~/projects/gpt-protein/data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa esm_token_embed_33/esm_33layer_split0.hdf5 --model 33 --maxlen 5800 --gpu 0 --pertoken
# python ~/projects/gpt-protein/bin/esm_sequence_embed.py ~/projects/gpt-protein/data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa esm_token_embed_33/esm_33layer_split1.hdf5 --model 33 --maxlen 5800 --gpu 1 --pertoken
# python ~/projects/gpt-protein/bin/esm_sequence_embed.py ~/projects/gpt-protein/data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa esm_token_embed_33/esm_33layer_split2.hdf5 --model 33 --maxlen 5800 --gpu 2 --pertoken
# python ~/projects/gpt-protein/bin/esm_sequence_embed.py ~/projects/gpt-protein/data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa esm_token_embed_33/esm_33layer_split3.hdf5 --model 33 --maxlen 5800 --gpu 3 --pertoken

# Commands for 33-layer model with 30 layer embedding
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split0.fa data/swissprot/esm_embed_33/esm_33layer_extract30_split0.hdf5 --model 33 --layer 30 --maxlen 5800 --gpu 0
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split1.fa data/swissprot/esm_embed_33/esm_33layer_extract30_split1.hdf5 --model 33 --layer 30 --maxlen 5800 --gpu 1
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split2.fa data/swissprot/esm_embed_33/esm_33layer_extract30_split2.hdf5 --model 33 --layer 30 --maxlen 5800 --gpu 2
# python bin/esm_sequence_embed.py data/swissprot/uniprot_sprot_split/uniprot_sprot_split3.fa data/swissprot/esm_embed_33/esm_33layer_extract30_split3.hdf5 --model 33 --layer 30 --maxlen 5800 --gpu 3

import os
import logging
import argparse

import torch

import h5py
from tqdm.auto import tqdm

from proteinclip import esm_wrapper, fasta_utils


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Run to query and cache ESM embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta", type=str, help="Fasta file to read.")
    parser.add_argument("out_h5", type=str, help="Output h5 file.")
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        choices=esm_wrapper.ESM_CALLABLES.keys(),
        default=33,
        help="Model to use.",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=0,
        help="Layer offset: 0 for last layer by default, otherwise provide an integer 0 < l <= model_size.",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=0,
        help="Maximum length to embed; longer sequences are skipped. 0 disables check.",
    )
    parser.add_argument("--pertoken", action="store_true", help="Embed per token.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU to use.")
    return parser


def main() -> None:
    """Run script."""
    parser = build_parser()
    args = parser.parse_args()
    assert args.out_h5.endswith(".hdf5")

    # auto-determine key function based on fasta file name
    bname = os.path.basename(args.fasta)
    if bname.startswith("cath-dataset-nonredundant"):
        key_func = lambda s: s.split("|")[2].split("/")[0]
    else:
        key_func = lambda s: s.split("|")[1] if "|" in s else s
    sequences = fasta_utils.read_fasta(args.fasta, key_func=key_func)

    if args.maxlen:
        logging.info(f"Dropping sequences longer than {args.maxlen} amino acids.")
        l = len(sequences)
        sequences = {k: v for k, v in sequences.items() if len(v) <= args.maxlen}
        logging.info(f"{len(sequences)} / {l} remain filtering length <= {args.maxlen}")

    logging.info(f"Embedding {len(sequences)} sequences.")

    with h5py.File(args.out_h5, "w") as sink:
        sink.attrs["model_size"] = args.model
        sink.attrs["length_limit"] = args.maxlen
        sink.attrs["fasta"] = args.fasta

        # Loop over and embed sequences
        for i, (k, v) in tqdm(
            enumerate(sequences.items()),
            total=len(sequences),
            desc=f"Embedding w/ {args.model}-layer model",
        ):
            sink.create_dataset(
                k,
                data=esm_wrapper.embed_sequence_single(
                    v,
                    model_size=args.model,
                    embed_layer=args.layer,
                    mean_across_seq=not args.pertoken,
                    device=torch.device(f"cuda:{args.gpu}"),
                ),
            )


if __name__ == "__main__":
    main()
