"""
Functions and classes for working with CATH data.
"""

from itertools import chain
import logging
from typing import Callable, Collection, Dict, Literal, Tuple

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from proteinclip.fasta_utils import read_fasta
from proteinclip.data_utils import DATA_DIR

CATH_DIR = DATA_DIR / "cath"
assert CATH_DIR.is_dir()


class CathSequences:
    """Class for working with CATH sequences."""

    def __init__(
        self,
        version: Literal["S20", "S40"] = "S20",
        cath_dir: str = CATH_DIR,
        blacklisted_identifiers: Collection[str] | None = None,
    ):
        self._cath_dir = cath_dir

        fa_files = [f for f in CATH_DIR.glob("*.fa") if version in f.name]
        assert len(fa_files) == 1
        self._cath_fa = str(fa_files.pop())

        # Load the sequences
        # An example fasta key: cath|current|1a48A01/2-112
        self._seq_records = read_fasta(
            self._cath_fa, key_func=lambda s: s.split("|")[2].split("/")[0]
        )
        if blacklisted_identifiers:
            logging.info(f"Excluding n={len(blacklisted_identifiers)} identifiers")
            self._seq_records = {
                k: v
                for k, v in self._seq_records.items()
                if k not in set(blacklisted_identifiers)
            }

        # Read in the table that maps these to CATH domains
        table_files = list(CATH_DIR.glob("cath-domain-list-*.txt"))
        assert len(table_files) == 1
        self._cath_table_file = str(table_files.pop())
        self._cath_table = pd.read_csv(
            self._cath_table_file,
            comment="#",
            delimiter="\s+",
            low_memory=False,
            header=None,
        )
        self._cath_table.columns = [
            "domain",
            "class",
            "architecture",
            "topology",
            "homologous_superfamily",
            "s35_cluster",
            "s60_cluster",
            "s95_cluster",
            "s100_cluster",
            "s100_count",
            "domain_length",
            "resolution",
        ]
        self._cath_table = self._cath_table[
            self._cath_table["domain"].isin(self._seq_records.keys())
        ]
        self._cath_table = self._cath_table.set_index("domain")
        assert len(self._cath_table) == len(self._seq_records)

        # Create a new column that combines the four CATH classifications
        self._cath_table["cath_classification"] = self._cath_table[
            ["class", "architecture", "topology", "homologous_superfamily"]
        ].apply(lambda x: ".".join([str(s) for s in x]), axis=1)

        # Report lengths
        logging.info(
            f"Domain lengths from {np.min(self._cath_table['domain_length'])} to {np.max(self._cath_table['domain_length'])}"
        )

        # Group by homologous_superfamily
        self._grouped = self._cath_table.groupby("cath_classification").groups
        self._multi_member_groups = {
            k: v for k, v in self._grouped.items() if len(v) > 1
        }
        logging.info(
            f"Found {len(self._multi_member_groups)} multi-member groups among {len(self._grouped)} total groups."
        )

    def get_sequence(self, identifier: str) -> str:
        """Get the sequence for the identifier if it exists; empty string otherwise."""
        return self._seq_records.get(identifier, "")

    def get_sequence_items(self) -> Collection[Tuple[str, str]]:
        """Return the sequence items as a collection of (identifier, sequence) tuples."""
        return list(self._seq_records.items())

    def __len__(self) -> int:
        return len(self._seq_records)

    def build_ref_query_set(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build a dataset to test retrieval, inspired by:
        https://www.frontiersin.org/articles/10.3389/fbinf.2022.1033775/full

        The reference dataset is all domains
        The query dataset is all domains that belong to a multi-member CATH group
        Return each as a dictionary of {domain: sequence} mappings.
        """
        multi_member_seqs = set(chain.from_iterable(self._multi_member_groups.values()))
        references = {k: self._seq_records[k] for k in self._seq_records.keys()}
        queries = {k: self._seq_records[k] for k in multi_member_seqs}
        assert set(queries.keys()) < set(references.keys())
        return references, queries

    def get_cath_class(self, domain: str) -> str:
        """Return the CATH class for a domain."""
        return self._cath_table.loc[domain]["cath_classification"]

    def cath_class_is_multidomain(self, cath_class: str) -> bool:
        """Return True if the CATH class is a multi-domain class."""
        return cath_class in self._multi_member_groups


def eval_embedding_function_on_dataset(
    embed_function: Callable[[str], np.ndarray],
    cath_sequences: CathSequences,
    func_on_identifier: bool = True,
) -> Dict[str, Dict[str, float | str | bool]]:
    """
    Evaluate an embedding function on a dataset of sequences.
    Return a dictionary of {seq_id: {seq_id: score}} mappings.
    """
    reference_sequences, query_sequences = cath_sequences.build_ref_query_set()
    ref_embeddings = {
        k: embed_function(v if not func_on_identifier else k)
        for k, v in tqdm(reference_sequences.items(), desc="Embedding references")
    }
    query_embeddings = {
        k: embed_function(v if not func_on_identifier else k)
        for k, v in tqdm(query_sequences.items(), desc="Embedding queries")
    }

    retval = {}
    for query, query_embeddings in tqdm(query_embeddings.items()):
        # Create a version of the reference embeddings without the query
        ref_keys = sorted(set(ref_embeddings.keys()) - {query})
        ref_embeddings_minus_query = np.array([ref_embeddings[k] for k in ref_keys])

        # Compute cosine similarity
        scores = np.dot(query_embeddings, ref_embeddings_minus_query.T)
        assert scores.ndim == 1
        max_idx = np.argmax(scores)
        matched_key = ref_keys[max_idx]

        # Get the class for both the query and the match
        query_class = cath_sequences.get_cath_class(query)
        match_class = cath_sequences.get_cath_class(matched_key)
        retval[query] = {
            "max_score": np.max(scores),
            "query_cath_class": query_class,
            "match_cath_class": match_class,
            "correct": query_class == match_class,
            "query_length": len(query_sequences[query]),
        }
    return retval


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    x = CathSequences()
    print(x.get_cath_class("1a48A01"))

    ref, query = x.build_ref_query_set()

    f = lambda x: np.random.normal(size=768)

    eval_results = eval_embedding_function_on_dataset(f, x)
    eval_accuracy = np.array([v["correct"] for v in eval_results.values()])
    print(f"Accuracy: {eval_accuracy.mean()}")
