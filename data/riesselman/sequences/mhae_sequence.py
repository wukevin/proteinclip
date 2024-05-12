"""Write the sequence of the Mhae gene to a file."""

import os
import pandas as pd


def write_mhae_sequence():
    # Read in the sequence
    source_fname = os.path.join(os.path.dirname(__file__), "pcbi.1004421.s003.xlsx")
    df = pd.read_excel(source_fname, sheet_name="G0")
    sequence_column = df.iloc[:, 1]
    sequence = "".join([s for s in sequence_column if not pd.isna(s) and s != "-"])
    # Should be comparable to https://www.uniprot.org/uniprotkb/P20589/entry#sequences
    sequence = "M" + sequence
    assert len(sequence) == 330

    outfile = os.path.join(os.path.dirname(__file__), "mhae_sequence_custom.fasta")
    with open(outfile, "w") as f:
        f.write(">custom|mhae_PMC4537296|pcbi.1004421.s003.xlsx\n")
        f.write(sequence + "\n")


if __name__ == "__main__":
    write_mhae_sequence()
