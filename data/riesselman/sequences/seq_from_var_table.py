"""
Load the sequence for the BG_STRSQ_hmmerbit experiment.

This sequence is contained in the mutational screen file since every position is
listed as a reference/alt pair, so we can reconstruct the reference in this way.

Handles:
- BG_STRSQ_hmmerbit
- HG_FLU_Bloom2016
- B3VI55_LIPSTSTABLE
"""

import os
from typing import Sequence

import pandas as pd


def mutants2wt(mutations: Sequence[str]) -> str:
    """Convert a set of mutations to the wildtype sequence."""
    seq = {1: "M"}
    for m in mutations:
        ref = m[0]
        _alt = m[-1]
        pos = int(m[1:-1])
        if pos in seq:
            assert seq[pos] == ref
        else:
            seq[pos] = ref
    return "".join([seq[i] for i in range(1, max(seq.keys()) + 1)])


def main():
    d = os.path.dirname(__file__)

    keys = [
        "BG_STRSQ_hmmerbit",
        "HG_FLU_Bloom2016",
        "B3VI55_LIPSTSTABLE",
        "B3VI55_LIPST_Whitehead2015",
    ]
    for k in keys:
        table = pd.read_excel(
            os.path.join(d, "../riesselman_mutations.xlsx"),
            sheet_name=k,
        )
        seq = mutants2wt(table["mutant"])

        with open(os.path.join(d, f"{k.lower()}.fasta"), "w") as sink:
            sink.write(f">custom|{k}|from_variant_table\n")
            sink.write(seq + "\n")


if __name__ == "__main__":
    main()
