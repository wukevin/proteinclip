"""Code to write"""

import os
import re
import json
import gzip
from collections import defaultdict
from pathlib import Path
import logging
from typing import *

from diskcache import FanoutCache

import numpy as np

from tqdm.auto import tqdm

from proteinclip import gpt
from proteinclip.fasta_utils import read_fasta, swissprot_identifier_human_only


cache = FanoutCache(
    directory=os.path.expanduser("~/.cache/uniprot"),
    timeout=0.1,
    size_limit=int(8e9),  # Size limit of cache in bytes; 8GB
    eviction_policy="least-recently-used",
)


class SwissProtDataReader:
    """Reads the SwissProt .dat file and provides access to metadata.

    Provides access to records via accesion, e.g. P38398."""

    # Functions that read in lines for the .dat file.
    line_headers = {
        "AC": lambda x: x.split(";")[0],  # Accession
        "GN": lambda x: x.split(" ")[0],  # Gene name
        "OS": lambda x: x.strip("."),  # Organism
        "OX": lambda x: x.split("=", 1)[-1].split(" ")[0].strip(";"),  # Organism ID
        "CC": lambda x: x,  # Comment
        "PE": lambda x: int(x.split(":")[0]),  # Evidence
    }

    def __init__(
        self,
        dat_fname: str | Path,
        fasta_fname: str | Path | None = None,
        fasta_seq_len_limit: int = 0,
        drop_no_function: bool = False,
    ):
        self.dat_fname = dat_fname
        self.records: Dict[str, Dict[str, Any]] = {}
        for accession, records in self._group():
            assert accession not in self.records, f"Duplicated accession: {accession}"
            # print(accession)
            # print(records)
            parsed = self._parse(records)
            self.records[accession] = parsed

        if fasta_fname:
            sequences = read_fasta(
                fasta_fname,
                key_func=lambda s: s.split("|")[1],
                length_limit=fasta_seq_len_limit,
            )
            accs_with_seq = set()
            for accession, sequence in sequences.items():
                if accession in self.records:
                    self.records[accession]["sequence"] = sequence
                    accs_with_seq.add(accession)
            accs_without_seq = set(self.records.keys()) - accs_with_seq
            if accs_without_seq:
                logging.warning(
                    f"{len(accs_without_seq)} records without sequence; dropping"
                )
                for k in accs_without_seq:
                    self.records.pop(k)

        if drop_no_function:
            orig_len = len(self.records)
            self.records = {k: v for k, v in self.records.items() if v["function"]}
            if (new_len := len(self.records)) < orig_len:
                logging.warning(
                    f"Dropped {orig_len - new_len} records without function texts."
                )

        self._organism_gene_accession_maps = {}

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.records[key]

    def __len__(self) -> int:
        return len(self.records)

    def get(self, key: str, default: Any = None) -> Dict[str, Any] | Any:
        return self.records.get(key, default)

    def keys(self) -> Iterator[str]:
        return self.records.keys()

    def items(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        return self.records.items()

    def values(self) -> Iterator[Dict[str, Any]]:
        return self.records.values()

    def unique_organisms(self, use_id: bool = False) -> List[str]:
        """Get the unique organisms in the dataset."""
        if use_id:
            k = "organism_id"
        else:
            k = "organism"
        return list(set(r[k] for r in self.records.values()))

    def gene2accession(self, species: str) -> Dict[str, List[str]]:
        """Build a dictionary of gene names to accessions for a given species."""
        retval = defaultdict(list)
        for k, v in self.records.items():
            if v["organism"] == species and v["name"]:
                retval[v["name"]].append(k)
        self._organism_gene_accession_maps[species] = retval
        return retval

    def query_gene_identifier(
        self, gene_name: str, species: str = "Homo sapiens (Human)"
    ) -> List[str]:
        """Query for identifiers corresponding to the gene name in organism."""
        if species not in self._organism_gene_accession_maps:
            self.gene2accession(species)  # Sets the mapping

        return self._organism_gene_accession_maps[species].get(gene_name, [])

    def get_description(self, identifier: str) -> str:
        """Returns the function description for the given identifier."""
        if not identifier:
            return ""
        entry = self.records.get(identifier, dict()).get("function", "")
        return entry

    def _read(self) -> Iterator[Tuple[str, str]]:
        """Reads the file and yields lines with keys and values."""
        opener = gzip.open if self.dat_fname.endswith(".gz") else open
        with opener(self.dat_fname, "rt") as source:
            for line in tqdm(source, desc=f"Reading {self.dat_fname}"):
                line = line.strip()
                if line == "//":
                    yield line, ""
                else:
                    parts = line.split(" ", maxsplit=1)
                    if len(parts) != 2:
                        continue
                    header_key, rest = parts
                    header_key = header_key.strip()
                    rest = rest.strip()

                    if header_key in self.line_headers:
                        yield header_key, self.line_headers[header_key](rest)

    def _group(self) -> Iterator[Tuple[str, Dict[str, List[str]]]]:
        """Group lines into records that share a common AC key."""
        curr_record = None
        curr_values = defaultdict(list)
        for k, v in self._read():
            if k == "AC":
                if curr_record:
                    if curr_values:
                        yield curr_record, curr_values
                    else:
                        # This is a case where we have two or more AC lines in a
                        # row; this indicates a lot of synonymous accessions, so
                        # we don't return until we actually build non-AC lines.
                        continue
                curr_record = v
                curr_values = defaultdict(list)
            else:
                curr_values[k].append(v)
        yield curr_record, curr_values

    @classmethod
    def _function_from_comments(
        cls, comments: List[str], strip_references: bool = True
    ) -> str:
        """Extract the function from the comments."""
        retval = []
        in_function = False
        for comment in comments:
            comment = comment.strip()
            if comment.startswith("-!- FUNCTION:"):
                in_function = True
                _, func_text = comment.split(":", maxsplit=1)
                retval.append(func_text.strip())
            elif in_function:
                # We have reached the end of the function region; break out
                if comment.startswith("-!-") or set(comment) == {"-"}:
                    break
                retval.append(comment)
        retval = " ".join(retval)
        if strip_references:
            # Remove references
            retval = re.sub(r"(ECO|PubMed|UniProtKB):[A-Z]?[0-9]+", "", retval)
            # Remove the delimiters and empty parentheses
            retval = retval.replace("|", "")
            retval = re.sub(r"\((, )*\)", "", retval)
            retval = re.sub(r"\{(, )*\}", "", retval)
            # Remove braces and trailing periods.
            retval = retval.replace(" .", ".").replace("..", ".")
        return retval

    @classmethod
    def _gene_name_from_comments(cls, comments: List[str]) -> str:
        for line in comments:
            tokens = line.split("=")
            if len(tokens) == 2:
                k, v = [t.strip() for t in tokens]
                if k == "Name":
                    return v.strip(";")
            else:
                continue
        return ""

    @classmethod
    def _parse(cls, records: Dict[str, List[str]]) -> Dict[str, str]:
        """Parse the records into a dictionary."""
        return {
            "organism": records["OS"].pop(),
            "organism_id": records["OX"].pop(),
            "name": cls._gene_name_from_comments(records["GN"]),
            "evidence": records["PE"].pop(),
            "function": cls._function_from_comments(records["CC"]),
        }


def strip_pubmed_references(text: str) -> str:
    """Remove references to PubMed articles from text."""
    stripped = re.sub(r"\(PubMed:\s+<a.*?</a>\s*\)", "", text)

    # Fix instances of long runs of whitespace caused after removing refs
    fix_whitespace = re.sub(r"\s{2,}", " ", stripped)

    # Fix " ." -> "."
    return re.sub(r"\s\.", ".", fix_whitespace)


def get_human_swissprot_ids(
    fname=os.path.expanduser("~/data/uniprot_sprot.fasta.gz"),
) -> List[str]:
    """Get all SwissProt IDs."""
    assert os.path.isfile(fname)

    seqs = read_fasta(
        fname,
        key_func=swissprot_identifier_human_only,
    )
    return [s for s in seqs.keys() if s]


def get_human_descriptions() -> Dict[str, str]:
    """Call the function fetcher for each uniprot identifier."""
    ids = get_human_swissprot_ids()
    retval = {}
    for i in (pbar := tqdm(ids)):
        pbar.set_description(f"Querying: {i}")
        func = fetch_uniprot_function(i)
        if not func:
            logging.warning("Empty function for %s", i)
        retval[i] = func
    return retval


def embed_function_descriptions(
    dat_fname: str | Path,
    human_only: bool = False,
    model: gpt.GPT_EMBED_MODELS = "text-embedding-ada-002",
    write_json: str = "",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Get embeddings corresponding to each human protein's function description

    If write_json is specified, write the text and embeddings to a JSON file as
    a "side-effect".
    """
    dat = SwissProtDataReader(dat_fname, **kwargs)
    if human_only:
        dat = {k: v for k, v in dat.items() if v["organism"] == "Homo sapiens (Human)"}
    function_texts = {k: v["function"] for k, v in dat.items()}
    embeddings = {
        i: (gpt.get_openai_embedding(q, model=model) if q else np.zeros(1536))
        for i, q in tqdm(function_texts.items(), desc=f"Embedding via {model}")
    }

    if write_json:
        with open(write_json, "w") as sink:
            obj = {
                k: {"text": function_texts[k], "embedding": embeddings[k].tolist()}
                for k in function_texts
            }
            json.dump(obj, sink, indent=2)

    return embeddings


if __name__ == "__main__":
    s = SwissProtDataReader(
        "/home/wukevin/projects/gpt-protein/data/swissprot/uniprot_sprot.dat.gz"
    )
    print(len(s.unique_organisms()))
    print(s.unique_organisms(use_id=True))
    # embed_function_descriptions(
    #     dat_fname="/home/wukevin/projects/gpt-protein/data/swissprot/uniprot_sprot.dat.gz",
    #     model="text-embedding-3-large",
    #     # write_json="swissprot_descriptions.json",
    # )
    # Example usage
    # uniprot_id = "P43403"
    # uniprot_id = "P05067"
    # uniprot_id = "A0A0C5B5G6"
    # uniprot_id = "foobarbazlolll"
    # uniprot_id = "Q15172"
    # uniprot_id = "O00237"
    # uniprot_id = "O14907"
    # uniprot_id = "O43598"
    # print(fetch_uniprot_function(uniprot_id))
