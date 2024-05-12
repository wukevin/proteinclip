"""
Commands for pre-computing mutation embeddings

Note: the 36 layer version is on sherlock, others are local
python ~/projects/gpt-protein/proteinclip/mut_data.py -m 6 -o riesselman_scores_esm2_6layer.hdf5 -g 0
python ~/projects/gpt-protein/proteinclip/mut_data.py -m 12 -o riesselman_scores_esm2_12layer.hdf5 -g 0
python ~/projects/gpt-protein/proteinclip/mut_data.py -m 30 -o riesselman_scores_esm2_30layer.hdf5 -g 1
python ~/projects/gpt-protein/proteinclip/mut_data.py -m 33 -o riesselman_scores_esm2_33layer.hdf5 -g 2
python ~/projects/gpt-protein/proteinclip/mut_data.py -m 36 -o riesselman_scores_esm2_36layer.hdf5 -g 0
python ~/projects/gpt-protein/proteinclip/mut_data.py -p t5xl_half -o riesselman_scores_prott5.hdf5 -g 3
"""

import argparse
import logging
import json
import gzip
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import h5py

from tqdm.auto import tqdm

from proteinclip import fasta_utils, esm_wrapper, prott5_wrapper

DATA_DIR = (Path(__file__) / "../../data").resolve()
assert DATA_DIR.is_dir(), f"Data directory not found: {DATA_DIR}"

Mutation = namedtuple("Mutation", ["ref_seq", "alt_seq", "score"])

UNIPROT_NAME_TO_ID = {
    "BLAT_ECOLX": "P62593",
    "DLG4_RAT": "P31016",
    "GAL4_YEAST": "P04386",
    "HSP82_YEAST": "P02829",
    "KKA2_KLEPN": "P00552",
    "PABP_YEAST": "P04147",
    "POLG_HCVJF": "Q99IB8",
    "RL401_YEAST": "P0CH09",  # Renamed RL40B_YEAST
    "UBE4B_MOUSE": "Q9ES00",
    "YAP1_HUMAN": "P46937",
    "AMIE_PSEAE": "P11436",
    "P84126_THETH": "P84126",
    "P84126_THETH (K207 (uniprot) -> F207 (paper))": "P84126",
    "TRPC_SULSO": "Q06121",  # Remapped to TRPC_SACS2
    "TRPC_THEMA": "Q56319",
    "TRPC_THEMA (C102 (uniprot) -> S102 (paper))": "Q56319",
    "IF1_ECOLI": "P69222",
    "MK01_HUMAN": "P28482",
    "RASH_HUMAN": "P01112",
    "RL401_YEAST": "P0CH09",
    "BRCA1_HUMAN": "P38398",
    "TPMT_HUMAN": "P51580",
    "PTEN_HUMAN": "P60484",
    "HIS7_YEAST": "P06633",
    "CALM1_HUMAN": "P0DP23",
    "TPK1_HUMAN": "Q9H3S4",
    "SUMO1_HUMAN": "P63165",
    "UBC9_HUMAN": "P63279",
    "P12497": "P12497",
    "B3VI55_LIPST": "B3VI55_LIPST_Whitehead2015",
    "(Stabilized sequence based on B3VI55_LIPST)": "B3VI55_LIPSTSTABLE",
}

PMID_TO_UNIPROT = {
    26040002: "BG_STRSQ_hmmerbit",  # CUSTOM ID, NOT UNIPROT
    26132554: "P15659",
    27271655: "HG_FLU_Bloom2016",
    26274323: "mhae_PMC4537296",  # CUSTOM ID, NOT UNIPROT
}

# Dataset names
DSET_NAMES_TO_UNIPROT = {
    "BG505_ENV": "BG505",
    "BF520_ENV": "BF520",
}

# Maps between the excel table and summary table
DATASET_NAME_MAP = {
    "BRCA1_HUMAN_BRCT": "BRCA_HUMAN_BRCT",
    "F7YBW7_MESOW_vae": "PARE_PARD",
    "RL401_YEAST_Fraser2016": "RL401_Fraser2016",
    "CALM1_HUMAN_Roth2017": "CALM1_HUMAN",
    "parEparD_Laub2015_all": "PARE_PARD",
    "UBC9_HUMAN_Roth2017": "UBC9_HUMAN",
    "SUMO1_HUMAN_Roth2017": "SUMO1_HUMAN",
    "PABP_YEAST_Fields2013-doubles": "PABP_singles,PABP_doubles",
    "PABP_YEAST_Fields2013-singles": "PABP_singles,PABP_doubles",
    "RASH_HUMAN_Kuriyan": "RASH_HUMAN",
    "BG_STRSQ_hmmerbit": "BG_STRSQ",
    "RL401_YEAST_Bolon2014": "RL401_Bolon2014",
    "MK01_HUMAN_Johannessen": "MK01_HUMAN",
    "HSP82_YEAST_Bolon2016": "HSP82",
    "YAP1_HUMAN_Fields2012-singles": "YAP1: Fields 2012",
    "BF520_env_Bloom2018": "BF520_ENV",
    "UBE4B_MOUSE_Klevit2013-singles": "UBE4B",
    "tRNA_mutation_effect": "TRNA_YEAST",
    "HG_FLU_Bloom2016": "HG_FLU",
    "B3VI55_LIPST_Whitehead2015": "B3VI55_LIPST",
    "TPK1_HUMAN_Roth2017": "TPK1_HUMAN",
    "GAL4_YEAST_Shendure2015": "GAL4",
    "TIM_SULSO_b0": "TIM_SULSO",
    "POLG_HCVJF_Sun2014": "POLG_HCVJF",
    "HIS7_YEAST_Kondrashov2017": "HIS7_YEAST",
    "TPMT_HUMAN_Fowler2018": "TPMT_HUMAN",
    "DLG4_RAT_Ranganathan2012": "DLG",
    "MTH3_HAEAESTABILIZED_Tawfik2015": "MTH3_HAEAESTABILIZED",
    "AMIE_PSEAE_Whitehead": "AMIE_PSEAE",
    "BRCA1_HUMAN_RING": "BRCA_HUMAN_RING",
    "P84126_THETH_b0": "TIM_THETH",
    "PTEN_HUMAN_Fowler2018": "PTEN_HUMAN",
    "IF1_ECOLI_Kishony": "IF1_ECOLI",
    "PA_FLU_Sun2015": "PA_FLU",
    "RL401_YEAST_Bolon2013": "RL401_Bolon2013",
    "KKA2_KLEPN_Mikkelsen2014": "KKA2",
    "TIM_THEMA_b0": "TIM_THEMA",
    "BG505_env_Bloom2018": "BG505_ENV",
}


def apply_mutation(mut: str, seq: str) -> str | None:
    """Apply mutation (e.g., "A123T") to the sequence, return new sequence."""
    if mut == "WT":
        return None
    pos = int(mut[1:-1])
    wt = mut[0]
    alt = mut[-1]
    if wt == alt or alt == "_" or wt == "_":
        return None
    # Position is 1-indexed; must be between [1, len(seq)]
    assert 0 < pos <= len(seq), f"Invalid position {pos} for length {len(seq)}"
    assert seq[pos - 1] == wt, f"Expected {wt} at position {pos} but got {seq[pos - 1]}"
    return seq[: pos - 1] + alt + seq[pos:]


def apply_multi_mutation(mut: str, seq: str) -> str | None:
    """Apply multiple mutations (e.g., "A123T:A124T") to the sequence, return new sequence."""
    for m in mut.split(":"):
        seq = apply_mutation(m, seq)
        if seq is None:
            return None
    return seq


def check_mutation(ref_seq: str, mut_string: str) -> bool:
    """Check that the mutation is valid; return False if not."""
    if mut_string == "WT":
        return False
    ref = mut_string[0]
    pos = int(mut_string[1:-1])
    mut = mut_string[-1]

    if ref == mut:
        return False

    # Length/indexing related
    if pos > len(ref_seq) or len(mut) != 1:
        return False

    # Sequence match
    if ref_seq[pos - 1] != ref:
        return False

    return True


def expand_mutations(
    mutation_tables: Dict[str, pd.DataFrame],
    summary_table: pd.DataFrame,
    sequences: Dict[str, str],
) -> Dict[str, List[Mutation]]:
    """Walk over the mutation tables and create mutation records."""
    retval = {}
    for sheet_name, df in (pbar := tqdm(mutation_tables.items())):
        pbar.set_description(sheet_name)
        sheet_mutations = []

        if sheet_name == "POL_HV1N5-CA_Ndungu2014":
            logging.warning(f"Creating manual metadata for {sheet_name}")
            # Manual entry - this is somehow missing in the summary table
            metadata = pd.Series(
                {
                    "Dataset Name(s)": "POL_HV1N5-CA_Ndungu2014",
                    "Uniprot ID (Nov 2015)": "P12497",
                    "PMID": 1,
                    "Protein name": "HIV-1 GAG-POL protein",
                    "Name(s) in Reference": "Rcmutant_RC_NL43",
                }
            )
        else:
            # Find the metadata for this current dataset in the metadata summary table
            metadata = summary_table.loc[
                summary_table["Dataset Name(s)"]
                == DATASET_NAME_MAP.get(sheet_name, sheet_name)
            ]
            if len(metadata) != 1:
                logging.warning(f"No metadata found for {sheet_name}")
                continue
            metadata = metadata.iloc[0]
        dset_name = metadata.get("Dataset Name(s)")

        # Find the UniProt ID for the protein using the metadata entries
        # First check presence in the Uniprot ID column, them PMIDs, then dataset names
        if (
            uniprot_identifier := metadata["Uniprot ID (Nov 2015)"].strip()
        ) in UNIPROT_NAME_TO_ID:
            uniprot_id = UNIPROT_NAME_TO_ID[uniprot_identifier]
        elif (pmid := int(metadata["PMID"])) in PMID_TO_UNIPROT:
            uniprot_id = PMID_TO_UNIPROT[pmid]
        elif dset_name in DSET_NAMES_TO_UNIPROT:
            uniprot_id = DSET_NAMES_TO_UNIPROT[dset_name]
        else:
            logging.warning(
                f"No UniProt ID found for {uniprot_identifier} / {pmid} in {sheet_name}"
            )
            continue
        full_seq = sequences.get(uniprot_id)
        assert full_seq, f"Could not find sequence for {uniprot_id} under {sheet_name}"

        # Handle cases where there is a change to the UniProt sequence
        if "(K207 (uniprot) -> F207 (paper))" in uniprot_identifier:
            if full_seq[206] == "F":
                pass
            else:
                assert (
                    full_seq[206] == "K"
                ), f"Expected K at 207 in {full_seq}, but got {full_seq[206]}"
                full_seq = full_seq[:206] + "F" + full_seq[207:]
        elif "(C102 (uniprot) -> S102 (paper))" in uniprot_identifier:
            assert full_seq[101] == "C"
            full_seq = full_seq[:101] + "S" + full_seq[102:]
        elif metadata["Protein name"].strip() == "Influenza polymerase PA subunit":
            assert full_seq[227] == "N"
            full_seq = full_seq[:227] + "K" + full_seq[228:]

        # Find the column for the experimental measurement
        meas_col = str(metadata["Name(s) in Reference"]).strip()
        if "," in meas_col:
            # Try to determine which measurement should be used based on the name
            name2meas = dict(
                zip(
                    metadata["Dataset Name(s)"].split(","),
                    metadata["Name(s) in Reference"].split(","),
                )
            )
            name2meas = {
                k.split("_")[-1].lower(): v.strip() for k, v in name2meas.items()
            }
            keyword = sheet_name.split("-")[-1].lower()
            meas_col = name2meas.get(keyword, None)
            assert (
                meas_col is not None
            ), f"Could not find measurement column for {sheet_name}: {name2meas}"
        assert (
            meas_col in df.columns
        ), f"Could not find column {meas_col} in {sheet_name}: \n {metadata}"

        # Expand each mutant
        for _i, row in df.iterrows():
            m, y = row["mutant"], row[meas_col]
            if pd.isna(y):
                continue

            try:
                if ":" in m:
                    mutated_seq = apply_multi_mutation(m, full_seq)
                else:
                    mutated_seq = apply_mutation(m, full_seq)
            except AssertionError as e:
                logging.critical(
                    f"Failed to apply mutation {m} to {uniprot_id} in {sheet_name}: {e}"
                )
                continue

            if mutated_seq:
                sheet_mutations.append(
                    Mutation(ref_seq=full_seq, alt_seq=mutated_seq, score=y)
                )
        retval[sheet_name] = sheet_mutations
    return retval


def load_riesselman_data(
    fname: str | Path = DATA_DIR / "riesselman/riesselman_mutations.xlsx",
    summary_fname: str | Path = DATA_DIR
    / "riesselman/riesselman_mutations_summary.xlsx",
    fasta_fname: str | Path = DATA_DIR / "swissprot/uniprot_sprot.fasta.gz",
    aux_fasta_dir: str | Path = DATA_DIR / "riesselman/sequences",
) -> Dict[str, List[Mutation]]:
    """Load the Rieselman data, returning as a dictionary where each key is an
    (experiment, measurement) and value is a list of (sprot_id, mutant, score)
    tuples.

    These are checked against fasta_fname swissprot sequences at loading time;
    consuming code can assume that the variant is correct."""
    fasta_key_fn = lambda s: s.split("|")[1]
    sequences = fasta_utils.read_fasta(str(fasta_fname), key_func=fasta_key_fn)
    for fasta_file in aux_fasta_dir.glob("*.fasta"):
        logging.info(f"Reading {fasta_file} for additional sequences.")
        sequences.update(fasta_utils.read_fasta(str(fasta_file), key_func=fasta_key_fn))

    summary_data = pd.read_excel(summary_fname, sheet_name="data")
    mutation_data = pd.read_excel(fname, sheet_name=None)

    return expand_mutations(mutation_data, summary_data, sequences)


def get_riesselman_references(mutations: Dict[str, List[Mutation]]) -> Dict[str, str]:
    """Get the reference wildtype sequence for each experiment."""
    retval = {}
    for exp, muts in mutations.items():
        m = next(iter(muts))
        retval[exp] = m.ref_seq
    return retval


def score_riesselman_data(
    data: Dict[str, List[Mutation]],
    scoring_func: Callable[[str, str], esm_wrapper.ScoredMissenseMutant],
    out_fname: str | Path,
) -> Dict[str, List[Dict[str, float | str | Tuple[float]]]]:
    """For each entry in the dataset, score all the associated mutants.

    Callable function to score should take in a mutant and a reference sequence,
    in that order.
    """
    assert out_fname.endswith(".hdf5") or out_fname.endswith(".h5")
    with h5py.File(out_fname, "w") as sink:
        for k, muts in data.items():
            for i, m in tqdm(enumerate(muts), desc=k, total=len(muts)):
                esm_results = scoring_func(m.alt_seq, m.ref_seq)
                sink.create_dataset(f"{k}/{i}/ref_seq", data=np.string_(m.ref_seq))
                sink.create_dataset(f"{k}/{i}/alt_seq", data=np.string_(m.alt_seq))
                sink.create_dataset(
                    f"{k}/{i}/ref_embed", data=esm_results.wildtype_mean_embed
                )
                sink.create_dataset(
                    f"{k}/{i}/alt_embed", data=esm_results.mutated_mean_embed
                )
                sink.create_dataset(f"{k}/{i}/exp_score", data=m.score)


def load_precomputed_scores(
    fname: str | Path,
    load_sequences: bool = True,
    experiments: Collection[str] | None = None,
) -> Generator[Tuple[str, str, str, np.ndarray, np.ndarray, float], None, None]:
    """Load precomputed data.

    Yields (experiment, ref_seq, alt_seq, ref_embed, alt_embed, exp_score) tuples.
    If load_sequences is turned off, ref_seq and alt_seq are empty strings for
    faster loading (~50% gain in throughput).
    """
    assert str(fname).endswith(".hdf5") or str(fname).endswith(".h5")

    with h5py.File(fname, "r") as source:
        for experiment in experiments or source.keys():
            records = source[experiment]
            logging.info(f"Loading {experiment}...")
            indices = sorted(map(int, records.keys()))
            for idx in (str(i) for i in indices):
                if load_sequences:
                    ref_seq = source[f"{experiment}/{idx}/ref_seq"][()].decode()
                    alt_seq = source[f"{experiment}/{idx}/alt_seq"][()].decode()
                else:
                    ref_seq, alt_seq = "", ""
                ref_embed = np.array(source[f"{experiment}/{idx}/ref_embed"])
                alt_embed = np.array(source[f"{experiment}/{idx}/alt_embed"])
                exp_score = np.array(source[f"{experiment}/{idx}/exp_score"]).item()
                yield (experiment, ref_seq, alt_seq, ref_embed, alt_embed, exp_score)


def build_parser() -> argparse.ArgumentParser:
    """Basic CLI parser."""
    parser = argparse.ArgumentParser(
        description="Score Riesselman data using ESM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "-m",
        "--model",
        type=int,
        choices=[36, 33, 30, 12, 6],
        help="ESM model size to run.",
    )
    model_group.add_argument(
        "-p",
        "--prott5",
        choices=prott5_wrapper.T5_MODEL_MAPPING.keys(),
        help="ProtT5 model to use; recommend t5xl_half.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="riesselman_scores.hdf5",
        help="Output h5 file to write.",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU to use.")
    return parser


def main() -> None:
    """Run as a script to generate Riesselman scores."""
    import torch

    args = build_parser().parse_args()
    data = load_riesselman_data()
    assert len(data) == 41  # There should be 41 mutants
    if args.model:
        logging.info(f"Using ESM model {args.model}.")
        score_riesselman_data(
            data,
            scoring_func=partial(
                esm_wrapper.eval_missense_mutant,
                model_size=args.model,
                device=torch.device(f"cuda:{args.gpu}"),
            ),
            out_fname=args.output,
        )
    else:
        logging.info(f"Using ProtT5 model {args.prott5}.")
        score_riesselman_data(
            data,
            scoring_func=partial(
                prott5_wrapper.eval_missense_mutant,
                model_key=args.prott5,
                device=torch.device(f"cuda:{args.gpu}"),
            ),
            out_fname=args.output,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
