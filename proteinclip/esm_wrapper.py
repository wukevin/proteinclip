import os
import json
import gzip
from functools import cache as fcache
from functools import lru_cache
import multiprocessing as mp
import argparse
from collections import namedtuple, ChainMap
import logging
from typing import *

import numpy as np
from torch.nn import functional as F
from tqdm.auto import tqdm

import torch
from torch import nn
import esm

from diskcache import FanoutCache

ScoredMissenseMutant = namedtuple(
    "ScoredMissenseMutant",
    [
        "mutated",  # Mutated amino acid sequence
        "wildtype",  # Wildtype amino acid sequence
        "mutated_mean_embed",  # Mean embedding of the mutated sequence
        "wildtype_mean_embed",  # Mean embedding of the wildtype sequence
        "weighted_mutated_mean_embed",  # Mean embedding of the mutated sequecne, weighted by relative change in each position
        "weighted_mean_embed_diff",  # Mean embedding difference, weighted by the relative change in each position
        "position_delta",  # Embedding delta at the position of the mutation
        "score",  # LLR of the mutant to the wildtype, given wildtype input
    ],
    defaults=["", "", 0, 0, 0, 0, 0, 0],
)

cache = FanoutCache(
    directory=os.path.expanduser("~/.cache/esm"),
    timeout=0.1,
    size_limit=int(12e9),  # Size limit of cache in bytes
    eviction_policy="least-recently-used",
)

eval_cache = FanoutCache(
    directory=os.path.expanduser("~/.cache/esm_full_eval_score"),
    timeout=0.1,
    size_limit=int(32e9),  # Size limit of cache in bytes
    eviction_policy="least-recently-used",
)

ESM_CALLABLES: Mapping[int, Callable] = {
    48: esm.pretrained.esm2_t48_15B_UR50D,
    36: esm.pretrained.esm2_t36_3B_UR50D,
    33: esm.pretrained.esm2_t33_650M_UR50D,
    30: esm.pretrained.esm2_t30_150M_UR50D,
    12: esm.pretrained.esm2_t12_35M_UR50D,
    6: esm.pretrained.esm2_t6_8M_UR50D,
}

DEVICE = torch.device(f"cuda:{torch.cuda.device_count() - 1}")

from proteinclip.fasta_utils import read_fasta, swissprot_identifier_human_only


@fcache
def get_model(model_size: int) -> Tuple[nn.Module, esm.Alphabet]:
    """Return model and alphabet for a given model size."""
    model, alphabet = ESM_CALLABLES[model_size]()
    model.eval()
    return model, alphabet


class CachedMissenseScores:
    """
    Manually defined cache for missense mutant scores. This is designed
    to be able to easily point at a set of files, load them all in parallel, and
    read from them as if they were a single dictionary. This is less robust than
    a dedicated cache solution, but is also easier to manually pre-compute.

    If keep_keys is specified, then keep only the keys that are also present in
    keep_keys. This is useful for reducing memory footprint.
    """

    def __init__(
        self,
        fnames: Sequence[str],
        n_threads: int = 1,
        keep_keys: Collection[Any] | None = None,
    ):
        self.fnames = fnames
        self.keep_keys = set(keep_keys) if keep_keys is not None else None
        for f in fnames:
            assert os.path.isfile(f), f"File {f} does not exist"

        if n_threads > 1:
            # This is faster but very memory intensive
            with mp.Pool(n_threads) as pool:
                results = list(pool.map(self._load_single, self.fnames))
        else:
            results = [self._load_single(f) for f in self.fnames]
        self.data = ChainMap(*results)

    def _passes_keep(self, k: Any) -> bool:
        """Determine if a key is present in set of specified keys to keep."""
        return k in self.keep_keys if self.keep_keys is not None else True

    def _load_single(self, fname: str) -> Dict[str, ScoredMissenseMutant]:
        logging.info(f"Reading protein cache: {fname}")
        opener = gzip.open if fname.endswith(".gz") else open
        with opener(fname, "rt") as f:
            # Load and convert the dictionary to named tuple
            retval = {
                k: ScoredMissenseMutant(
                    mutated=v["mutated"],
                    wildtype=v["wildtype"],
                    mutated_mean_embed=np.array(v["mutated_mean_embed"]),
                    wildtype_mean_embed=np.array(v["wildtype_mean_embed"]),
                    position_delta=np.array(v["position_delta"]),
                    score=v["score"],
                )
                for k, v in json.load(f).items()
                if v is not None and self._passes_keep(k)
            }
        return retval

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> ScoredMissenseMutant:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> ScoredMissenseMutant:
        """Get with a default value."""
        return self.data.get(key, default)

    def items(self) -> Iterable[Tuple[str, ScoredMissenseMutant]]:
        """Iterate over items."""
        return self.data.items()

    def keys(self) -> Iterable[str]:
        """Iterate over keys."""
        return self.data.keys()

    def values(self) -> Iterable[ScoredMissenseMutant]:
        """Iterate over values."""
        return self.data.values()


def embed_sequences(
    id_seq: List[Tuple[str, str]],
    batch_size: int = 16,
    model_size: int = 30,
    pbar: bool = True,
) -> Dict[str, np.ndarray]:
    """Embed a single sequence using ESM2 by averaging all tokens."""
    # The batch converter expects Sequence[Tuple[str, str]] where the first str
    # in the tuple is the sequence identifier and the second is the sequence
    # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L253
    indices = np.arange(0, len(id_seq), batch_size)

    model, alphabet = get_model(model_size)
    batch_converter = alphabet.get_batch_converter()

    labels = []
    sequence_representations = []
    m = model.to(DEVICE)
    for start_idx in tqdm(indices, disable=not pbar):
        end_idx = start_idx + batch_size
        batch_labels, _batch_strs, batch_tokens = batch_converter(
            id_seq[start_idx:end_idx]
        )
        labels.extend(batch_labels)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = m(
                batch_tokens.to(DEVICE), repr_layers=[33], return_contacts=False
            )
        token_representations = results["representations"][33]
        for i, tokens_len in enumerate(batch_lens):
            rep = token_representations[i, 1 : tokens_len - 1].cpu().numpy().mean(0)
            sequence_representations.append(rep)
    return dict(zip(batch_labels, sequence_representations))


@lru_cache(maxsize=16, typed=True)
def embed_sequence_single(
    sequence: str,
    model_size: int = 30,
    embed_layer: int = 0,
    mean_across_seq: bool = True,
    device: torch.device = DEVICE,
    return_logits: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Embed a single sequence using ESM2 by averaging all tokens.

    Embed layer indicates the layer to draw embedding from; 0 indicates last
    layer, otherwise provide a positive integer that is <= model size. Note that
    model size and layer numbering is 1-indexed, so the 30-th layer is accessed
    by "30". Note that the returned array may not have unit norm, and different
    layers may have very different norms!

    Returned embeddings have their special tokens removed, as is the case for
    returned logits (returned when return_logits is True).
    """
    if embed_layer == 0:
        embed_layer = model_size
    assert embed_layer > 0, "Embed layer must be > 0"
    assert embed_layer <= model_size, "Embed layer must be <= model size"
    model, alphabet = get_model(model_size)
    batch_converter = alphabet.get_batch_converter()

    _, _, tokens = batch_converter([("", sequence.replace("*", "<mask>"))])

    m = model.to(device)
    with torch.no_grad():
        results = m(tokens.to(device), repr_layers=[embed_layer], return_contacts=False)
    token_representations = results["representations"][embed_layer]
    # Indexing removes the BOS/EOS special tokens from the average
    rep = token_representations[0, 1 : 1 + len(sequence)].cpu().numpy()
    rep = rep.mean(0) if mean_across_seq else rep

    if return_logits:
        return rep, results["logits"][0, 1 : 1 + len(sequence)].cpu().numpy()
    return rep


def eval_missense_mutant(
    mutant_seq: str,
    wt_seq: str,
    model_size: int = 30,
    embed_layer: int = 0,
    device: torch.device = DEVICE,
) -> ScoredMissenseMutant | None:
    """Score the mutant using the given size of model.

    If embed_layer is 0 (default), use the last embedding layer; otherwise, use
    1-indexed layer indicated by embed_layer."""
    if not len(mutant_seq) == len(wt_seq):
        logging.warning("Sequences must be the same length")
        return None
    l: int = len(mutant_seq)
    if embed_layer == 0:
        embed_layer = model_size
    assert 0 < embed_layer <= model_size, "Embed layer must be <= model size"

    _model, alphabet = get_model(model_size)
    batch_converter = alphabet.get_batch_converter()

    # Replacing * with <mask> is done here such as not to affect the original
    # sequence, which we need to compute the LLR
    _, _, mut_tokens = batch_converter([("", mutant_seq.replace("*", "<mask>"))])
    _, _, wt_tokens = batch_converter([("", wt_seq.replace("*", "<mask>"))])

    # Since change is checked after tokenization, it includes the BOS/EOS tokens
    changed_indices: List[int] = torch.where(mut_tokens != wt_tokens)[1].tolist()
    if not changed_indices:
        return None
    if not (len(changed_indices) == 1):
        changed_index = None
    else:
        changed_index = changed_indices[0]

    # Embed the sequences
    wt_embed, wt_logits = embed_sequence_single(
        wt_seq,
        model_size=model_size,
        embed_layer=embed_layer,
        mean_across_seq=False,
        device=device,
        return_logits=True,
    )
    mut_embed = embed_sequence_single(
        mutant_seq,
        model_size=model_size,
        embed_layer=embed_layer,
        mean_across_seq=False,
        device=device,
        return_logits=False,
    )
    assert wt_embed.ndim == mut_embed.ndim == 2
    assert wt_embed.shape[0] == l

    # Calculate the LLR of the mutant to the reference.
    # Changed index is offset by 1 to remove the BOS token from indexing
    if changed_index is not None:
        # To get mutant and wt alphabet indices, we need to exclude special tokens
        mutant_idx = alphabet.get_idx(mutant_seq[changed_index - 1])
        wt_idx = alphabet.get_idx(wt_seq[changed_index - 1])
        assert (
            mutant_idx != wt_idx
        ), f"Mutant and wildtype are the same: {mutant_idx} {wt_idx} {mutant_seq[changed_index - 1]} {wt_seq[changed_index - 1]}"

        # Get the log probabilities
        # Logits are returned without special tokens; offset back by 1
        log_probs = F.log_softmax(
            torch.from_numpy(wt_logits[changed_index - 1]), dim=-1
        )
        score = (log_probs[mutant_idx] - log_probs[wt_idx]).item()
        position_delta = mut_embed[changed_index - 1] - wt_embed[changed_index - 1]
    else:
        score = np.nan
        position_delta = np.zeros(mut_embed.shape[-1])

    # Calculate the weighted mean embedding difference
    diff_rep = mut_embed - wt_embed  # This is (L x D)
    # Mean across the sequence, weighted by the relative change in each position
    # The relative change is given by the L2 norm of the difference at each position
    diff_norms = np.linalg.norm(diff_rep, axis=-1)
    assert diff_norms.shape == (l,), f"Invalid shape: {diff_norms.shape}"
    weighted_diff = np.average(diff_rep, axis=0, weights=diff_norms)
    assert weighted_diff.shape == (mut_embed.shape[-1],)

    return ScoredMissenseMutant(
        mutated=mutant_seq,
        wildtype=wt_seq,
        mutated_mean_embed=tuple(mut_embed.mean(0).tolist()),
        weighted_mutated_mean_embed=tuple(
            np.average(mut_embed, axis=0, weights=diff_norms).tolist()
        ),
        wildtype_mean_embed=tuple(wt_embed.mean(0).tolist()),
        weighted_mean_embed_diff=tuple(weighted_diff.tolist()),
        position_delta=tuple(position_delta.tolist()),
        score=score,
    )


def get_swissprot_human_sequence_embeddings(length_limit: int = 5800):
    # Official swissprot embeddings with T5 cap at 12k length limit
    # https://www.uniprot.org/help/embeddings
    # For esm2_t33_650M_UR50D, sequence of 5900 fits on 11GB GPU, and we scale
    # down a bit to 5800 as buffer
    sequences = read_fasta(
        os.path.expanduser("~/data/uniprot_sprot.fasta.gz"),
        key_func=swissprot_identifier_human_only,
        length_limit=length_limit,
    )

    embeddings = [embed_sequence_single(s) for s in tqdm(sequences.values())]
    return dict(zip(sequences.keys(), embeddings))


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_fname",
        type=str,
        help="Path to JSON file containing Mapping[identifier: (mutant, wildtype)]",
    )
    parser.add_argument(
        "-o",
        "--out_json",
        type=str,
        help="Path to output JSON file containing Mapping[identifier: ScoredMissenseMutant]",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=30,
        help="Number of layers in ESM2 model to use.",
    )
    parser.add_argument(
        "--takeevery",
        type=int,
        nargs=2,
        default=(1, 0),
        help="(x, y): Process items that are (i mod x) == y",
    )
    parser.add_argument(
        "-l",
        "--excludelen",
        type=int,
        default=0,
        help="If nonzero, exclude sequences of length > val. 5800 / 33-layer and 6600 / 30-layer for 11GB GPU memory",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode - run only a few inputs."
    )
    return parser


def main():
    """Run as script to pre-compute embeddings and scores for missense mutants."""
    parser = build_parser()
    args = parser.parse_args()

    # Load inputs
    opener = gzip.open if args.json_fname.endswith(".gz") else open
    with opener(args.json_fname, "rt") as f:
        inputs = json.load(f)
    logging.info(f"Loaded {len(inputs)} inputs")

    if args.takeevery[0] != 1:
        assert args.takeevery[1] < args.takeevery[0], f"Invalid combo: {args.takeevery}"
        inputs = {
            k: v
            for i, (k, v) in enumerate(inputs.items())
            if (i % args.takeevery[0]) == args.takeevery[1]
        }
        logging.info(f"Filtered down to {len(inputs)} inputs ({args.takeevery})")
    if args.debug:  # Limit to 10 inputs
        inputs = {k: v for i, (k, v) in enumerate(inputs.items()) if i < 10}
        logging.warning(f"Running debug mode with {len(inputs)} inputs")

    # Run
    results = {}
    for identifier, (mutant, wildtype) in tqdm(inputs.items()):
        # Drop on length if specified
        if args.excludelen > 0 and len(mutant) > args.excludelen:
            logging.warning(f"Skipping {identifier} due to length > {args.excludelen}")
            continue

        # Run through model
        result = eval_missense_mutant(
            mutant,
            wildtype,
            model_size=args.size,
            device=torch.device(f"cuda:{args.gpu}"),
        )
        results[identifier] = result._asdict() if result is not None else None

    # Write results
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    # x = eval_missense_mutant("RKDES", "RKEES", model_size=33)
    # print(np.array(x.mutated_mean_embed))
    # print(np.array(x.weighted_mutated_mean_embed))

    # Test different sequence lengths
    # 30-layer model handles up to 6600
    # 33-layer model handles up to 5800
    # seq = "R" * 6600
    # seq_mut = seq[:100] + "K" + seq[101:]
    # x = eval_missense_mutant(seq_mut, seq)
