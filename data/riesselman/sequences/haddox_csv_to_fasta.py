"""
Script to convert the Haddox et al. 2018 CSV file to a FASTA file.

The CSV files describe sequences BF520 and BG505. We write the sequences to a 
fasta file with key as
>custom|BF520|elife-34420-fig5-data2
>custom|BG505|elife-34420-fig4-data2
to ensure compatibility with other parsers.
"""

import os
import pandas as pd

ID2KEY = {
    "BF520": "custom|BF520|elife-34420-fig5-data2",
    "BG505": "custom|BG505|elife-34420-fig4-data2",
}


def extract_seq(fname: str) -> str:
    df = pd.read_csv(fname)
    return "".join(df["wildtype"])


def main():
    d = os.path.dirname(__file__)

    seqs = {
        "BG505": extract_seq(os.path.join(d, "elife-34420-fig4-data2.csv")),
        "BF520": extract_seq(os.path.join(d, "elife-34420-fig5-data2.csv")),
    }

    for k, v in seqs.items():
        with open(os.path.join(d, f"{k}.fasta"), "w") as f:
            f.write(f">{ID2KEY[k]}\n")
            f.write(v + "\n")


if __name__ == "__main__":
    main()
