"""
Code for embedding sequences in ProtT5 model (various versions):

Model listing: 
- prot_t5_xl_half_uniref50-enc:
  https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
  This version of the original ProtT5-XL-UniRef50 is mostly meant for conveniently
  creating amino-acid or protein embeddings with a low GPU-memory footprint without
  any measurable performance-decrease in our experiments. This model is fully usable
  on 8 GB of video RAM.

On the half-precision xl model, sequence length is capped at 3950 for 11GB GPUs.

Code here is significantly inspired by:
https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
"""

import os
import argparse
import logging
import re
from typing import Tuple

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer

from tqdm.auto import tqdm

from proteinclip import fasta_utils

MODEL_MAPPING = {
    "t5xl": "Rostlab/prot_t5_xl_uniref50",
    "t5xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
}


def sanitize_sequence(sequence: str) -> str:
    """Replace uncommon amino acids with X and insert whitespace."""
    retval = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    return retval


def get_model(
    model_key: str = "t5xl_half", device: torch.device = torch.device("cuda:0")
) -> Tuple[T5EncoderModel, T5Tokenizer]:
    """Get ProtT5 model and tokenizer."""
    model_name = MODEL_MAPPING[model_key]
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Run to query and cache ProtT5 embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta", type=str, help="Fasta file to read.")
    parser.add_argument("out_h5", type=str, help="Output h5 file.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_MAPPING.keys(),
        default="t5xl_half",
        help="Model to use.",
    )
    parser.add_argument(
        "-l",
        "--maxlen",
        type=int,
        default=0,
        help="Maximum length to embed; longer sequences are skipped.",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU to use.")
    return parser


def main():
    args = build_parser().parse_args()
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

    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer = get_model(model_key=args.model, device=device)

    with h5py.File(args.out_h5, "w") as sink, torch.no_grad():
        sink.attrs["model"] = args.model
        sink.attrs["length_limit"] = args.maxlen
        sink.attrs["fasta"] = args.fasta
        # Adds an additional EOS token to the end of the sequence
        for i, (identifier, seq) in tqdm(
            enumerate(sequences.items()),
            total=len(sequences),
            desc=f"Embedding with {args.model}",
        ):
            ids = tokenizer(
                sanitize_sequence(seq), add_special_tokens=True, padding="longest"
            )
            input_ids = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            embedding_repr = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )

            # Get the mean
            mean_repr = embedding_repr.last_hidden_state[0, : len(seq)].mean(dim=0)
            sink.create_dataset(identifier, data=mean_repr.cpu().numpy())


if __name__ == "__main__":
    main()
