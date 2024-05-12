import logging
from functools import lru_cache
import re
from functools import cache as fcache
from typing import *

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from proteinclip.esm_wrapper import ScoredMissenseMutant

T5_MODEL_MAPPING = {
    "t5xl": "Rostlab/prot_t5_xl_uniref50",
    "t5xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
}


@fcache
def get_model(
    model_key: str = "t5xl_half", device: torch.device = torch.device("cuda:0")
) -> Tuple[T5EncoderModel, T5Tokenizer]:
    """Get ProtT5 model and tokenizer."""
    model_name = T5_MODEL_MAPPING[model_key]
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer


def sanitize_sequence(sequence: str) -> str:
    """Replace uncommon amino acids with X and insert whitespace."""
    retval = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    return retval


@lru_cache(maxsize=16)
def embed_sequence_single(
    sequence: str,
    model_key: Literal["t5xl", "t5xl_half"] = "t5xl_half",
    mean_across_seq: bool = True,
    device: torch.device = torch.device("cuda:0"),
) -> np.ndarray:
    """Embed the sequence, return either l x d matrix or d-dimensional vector."""
    l: int = len(sequence)
    sequence = sanitize_sequence(sequence)
    model, tokenizer = get_model(model_key, device)

    # Adds a single special token at the end
    tok = tokenizer([sequence], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tok["input_ids"]).to(device)
    attn_mask = torch.tensor(tok["attention_mask"]).to(device)

    # Run through model
    with torch.no_grad():
        outputs = (
            model(input_ids=input_ids, attention_mask=attn_mask)
            .last_hidden_state.cpu()
            .numpy()
        )
    assert outputs.ndim == 3 and outputs.shape[0] == 1, f"Bad shape: {outputs.shape}"
    assert outputs.shape[1] == l + 1, f"Bad shape for sequence of {l}: {outputs.shape}"
    output = outputs.squeeze(0)[:l]  # (l x d), excludes last EOS token
    if mean_across_seq:
        output = output.mean(axis=0)
    return output


def eval_missense_mutant(
    mutant_seq: str,
    wt_seq: str,
    model_key: str = "t5xl_half",
    device: torch.device = torch.device("cuda:0"),
) -> ScoredMissenseMutant | None:
    """Score the given mutant."""
    # https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
    mutant_repr = embed_sequence_single(
        mutant_seq, model_key=model_key, mean_across_seq=True, device=device
    )
    wt_repr = embed_sequence_single(
        wt_seq, model_key=model_key, mean_across_seq=True, device=device
    )

    return ScoredMissenseMutant(
        mutated=mutant_seq,
        wildtype=wt_seq,
        mutated_mean_embed=tuple(mutant_repr.tolist()),
        wildtype_mean_embed=tuple(wt_repr.tolist()),
    )


if __name__ == "__main__":
    x = eval_missense_mutant("RKDESK", "RKDDSK")
    # print(x)
