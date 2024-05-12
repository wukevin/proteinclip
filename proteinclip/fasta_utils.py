"""
Code to work with fasta sequence files
"""

import os
import argparse
import logging
import gzip
from typing import Dict, Callable, Tuple

import numpy as np
from tqdm.auto import tqdm
from biotite.sequence import ProteinSequence, NucleotideSequence, CodonTable
from Levenshtein import distance as ldist

from proteinclip import gpt

logging.basicConfig(level=logging.INFO)

# https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html
AA_ONE_TO_THREE_LETTER = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "U": "Sec",  # Special
    "O": "Pyl",  # Special
}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of the given nucleotide sequence."""
    complements = {
        "A": "T",
        "T": "A",
        "G": "C",
        "C": "G",
        "N": "N",
    }
    return "".join([complements[nt] for nt in seq[::-1]])


def translate(nt_seq: str) -> str:
    """Translate the given nucleotide sequence."""
    return str(
        NucleotideSequence(nt_seq).translate(
            complete=True, codon_table=CodonTable.default_table()
        )
    )


def swissprot_identifier(s: str) -> str:
    """Extract the swissprot identifier from a swissprot fasta key."""
    return s.split("|")[1]


def swissprot_identifier_human_only(s: str) -> str:
    """Extract swissprot identifier if the sequence belongs to human otherwise empty string."""
    if "OS=Homo sapiens" in s:
        return swissprot_identifier(s)
    return ""


def read_fasta(
    file_path: str,
    key_func: Callable[[str], str] | None = None,
    length_limit: int = 0,
    disable_pbar: bool = True,
    **kwargs,
) -> Dict[str, str]:
    """Read the fasta file as a mapping from identifiers to sequences.

    If key_func is given, it should be a function that takes in an identifier and either
    returns a trimmed/cleaned version of the identiifer, or if the identifier should be
    dropped, then it should return an empty string.
    """
    sequences = {}
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as f:
        seq_id = None
        seq = []
        for line in tqdm(f, disable=disable_pbar):
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    assert seq_id not in sequences, f"Duplicated identifier: {seq_id}"
                    sequences[seq_id] = "".join(seq)
                seq_id = line[1:]
                if key_func:
                    seq_id = key_func(seq_id)
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            sequences[seq_id] = "".join(seq)
    logging.info(f"Read {len(sequences)} sequences from {file_path}")
    if length_limit > 0:
        sequences = {k: v for k, v in sequences.items() if len(v) <= length_limit}
        logging.info(f"{len(sequences)} remain filtering length <= {length_limit}")
    return sequences


def make_sequence_query(seq: str, expand_three_letter: bool = False, **kwargs) -> str:
    """Make a sequence query for embedding. Expects one-letter string.

    If expand_three_letter is True, then expand the sequence to use 3-letter
    codes separated by spaces. For example, RKDES > Arg Lys Asp Glu Ser."""
    if expand_three_letter:
        first_cap = lambda s: s[0].upper() + s[1:].lower()
        seq = " ".join([first_cap(AA_ONE_TO_THREE_LETTER[aa]) for aa in seq])
    return f"Amino acid sequence: {seq}"


def embed_sequences(fasta_file: str, **kwargs) -> Dict[str, np.ndarray]:
    """Embed the sequences in the fasta file."""
    seqs = read_fasta(fasta_file, **kwargs)
    queries = {i: make_sequence_query(s, **kwargs) for i, s in seqs.items()}

    embeddings = {i: gpt.get_openai_embedding(q) for i, q in tqdm(queries.items())}
    return embeddings


def find_most_similar(s: str, sequences: Dict[str, str]) -> Tuple[str, int]:
    """Find most similar sequence in the given dictionary of (identifer, sequence).

    Returns the top hit as a tuple of (identifier, distance).
    """
    min_dist = len(s)
    min_id = None
    for identifier, seq in sequences.items():
        d = ldist(s, seq, score_cutoff=min_dist)
        if d < min_dist:
            min_dist = d
            min_id = identifier
    assert min_id
    return min_id, min_dist


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Splitting a fasta file into multiple ~equally sized files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fastafile", type=str, help="Swissprot fasta file to read.")
    parser.add_argument(
        "-o", "--outdir", required=True, type=str, help="Output directory."
    )
    parser.add_argument(
        "-n", "--num", type=int, required=True, help="Number of splits."
    )
    return parser


def main():
    """Run as a script splitting the fasta file into multiple equally sized files."""
    args = build_parser().parse_args()

    seqs = read_fasta(args.fastafile)

    splits = [{} for _ in range(args.num)]
    for i, (k, v) in enumerate(seqs.items()):
        splits[i % args.num][k] = v
    print([len(s) for s in splits], sum(len(s) for s in splits))

    for i, split in enumerate(splits):
        bname = os.path.basename(args.fastafile).split(".")[0]
        with open(os.path.join(args.outdir, bname + f"_split{i}.fa"), "w") as f:
            for k, v in split.items():
                f.write(f">{k}\n{v}\n")


if __name__ == "__main__":
    main()
