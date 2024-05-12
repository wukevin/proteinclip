"""
Utilites for data loading and processing
"""

from pathlib import Path
import logging
from collections import ChainMap
from typing import *

import numpy as np
from tqdm.auto import tqdm
import h5py

import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).parent.parent / "data"
assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"


class CLIPDataset(Dataset):
    """Dataset for CLIP.

    Input can either be a list of pairs, where the first item corresponds to the
    first map, and second item to the second map, or can be a list of single keys
    that are present in both maps."""

    def __init__(
        self,
        pairs: List[Tuple[Any, Any]] | List[Any],
        map1: Mapping[Any, np.ndarray],
        map2: Mapping[Any, np.ndarray],
        enforce_unit_norm: bool = False,  # False for backwards compatibility
    ):
        self.pairs = pairs
        self.map1 = map1
        self.map2 = map2
        # If specfied, unit norm all vectors before returning in __getitem__
        self.enforce_unit_norm = enforce_unit_norm

        # Trim out the pairs where either of the two members is either
        # - not in the map
        # - maps to a zero vector
        orig_len = len(self.pairs)
        self.pairs = []
        for item in tqdm(pairs, desc="Checking for missing/zero embeddings"):
            if isinstance(item, (tuple, list)):
                item1, item2 = item
            else:
                item1 = item
                item2 = item
            if np.allclose(self.map1.get(item1, 0), 0) or np.allclose(
                self.map2.get(item2, 0), 0
            ):
                continue
            self.pairs.append(item)

        logging.info(
            f"Trimmed {orig_len - len(self.pairs)} pairs with missing/zero embeddings, {len(self.pairs)} remain."
        )

    def unique_vectors_map1(
        self, as_dict: bool = False
    ) -> Dict[Any, np.ndarray] | np.ndarray:
        """Get the unique vectors, return shape of (n, embed_dim)."""
        # Gather all the vectors present in the map and covered by the "pairs"
        if isinstance(next(iter(self.pairs)), (tuple, list)):
            unique_keys = sorted(set([x for x, _ in self.pairs]))
            if as_dict:
                return {k: self.map1[k] for k in unique_keys}
            return np.array([self.map1[k] for k in unique_keys])
        # We don't have unique keys to rely on; instead use identity comparisons
        # between vectors
        if as_dict:
            raise NotImplementedError("as_dict=True not implemented for this case")
        keys = [
            item[0] if isinstance(item, (tuple, list)) else item for item in self.pairs
        ]
        arr = np.array([self.map1[k] for k in keys])
        return array_unique_rows(arr)

    def unique_vectors_map2(
        self, as_dict: bool = False
    ) -> Dict[Any, np.ndarray] | np.ndarray:
        """Get the unique vectors, return shape of (n, embed_dim)."""
        if isinstance(next(iter(self.pairs)), (tuple, list)):
            unique_keys = sorted(set([x for _, x in self.pairs]))
            if as_dict:
                return {k: self.map2[k] for k in unique_keys}
            return np.array([self.map2[k] for k in unique_keys])
        # Gather all the vectors present in the map and covered by the "pairs"
        if as_dict:
            raise NotImplementedError("as_dict=True not implemented for this case")
        keys = [
            item[1] if isinstance(item, (tuple, list)) else item for item in self.pairs
        ]
        arr = np.array([self.map2[k] for k in keys])
        return array_unique_rows(arr)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        item = self.pairs[index]
        if len(item) == 2:
            item1, item2 = item
        else:
            item1 = item
            item2 = item

        embed1 = np.array(self.map1[item1]).astype(np.float32)
        embed2 = np.array(self.map2[item2]).astype(np.float32)
        if self.enforce_unit_norm:
            embed1 /= np.linalg.norm(embed1)
            embed2 /= np.linalg.norm(embed2)

        return {"x_1": embed1, "x_2": embed2}


class CLIPDataset2D1D(Dataset):
    """Dataset where the first item is (L x D) and second item is (D,).

    The return value is two tensors:
    - For the first item, it is padded to pad_to + 1 length, where the first token
      is the mean of the embeddings along the length dimension.
    - For the second item, return the D dimensional vector.
    """

    def __init__(
        self,
        pairs: List[Tuple[Any, Any]] | List[Any],
        map1: Mapping[Any, np.ndarray],
        map2: Mapping[Any, np.ndarray],
        pad_to: int = 5800,  # Only applies to map1; returned values are (pad_to + 1, D)
    ):
        self.pairs = pairs
        self.map1 = map1
        self.map2 = map2
        self.pad_to = pad_to

        # Trim out the pairs where either of the two members is either
        # - not in the map
        # - maps to a zero vector
        orig_len = len(self.pairs)
        self.pairs = []
        for i, item in enumerate(
            tqdm(pairs, desc="Checking for missing/zero embeddings")
        ):
            if isinstance(item, (tuple, list)):
                item1, item2 = item
            else:
                item1 = item
                item2 = item
            if np.allclose(self.map1.get(item1, 0), 0) or np.allclose(
                self.map2.get(item2, 0), 0
            ):
                continue
            self.pairs.append(item)

        logging.info(
            f"Trimmed {orig_len - len(self.pairs)} pairs with missing/zero embeddings, {len(self.pairs)} remain."
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index) -> Any:
        item = self.pairs[index]
        if len(item) == 2:
            item1, item2 = item
        else:
            item1 = item
            item2 = item

        embed1_raw = np.array(self.map1[item1]).astype(np.float32)  # L x D
        assert embed1_raw.ndim == 2, f"Expected 2D embedding, got {embed1_raw.ndim}"
        embed2 = np.array(self.map2[item2]).astype(np.float32)

        # Pad the first embedding to pad_to, set the corresponding tokens
        embed1 = np.zeros((self.pad_to + 1, embed1_raw.shape[1]), dtype=np.float32)
        embed1[0] = embed1_raw.mean(axis=0)
        embed1[1 : 1 + embed1_raw.shape[0]] = embed1_raw

        return {"x_1": embed1, "x_2": embed2}


class LabeledPairDataset(Dataset):
    """Dataset of pairs of (a, b, label) with embeddings for a and b."""

    def __init__(
        self,
        pairs: List[Tuple[Any, Any, float]],
        mapping: Mapping[Any, np.ndarray],
        strategy: Literal["concat", "diff", "sum", "separate"] = "concat",
    ):
        orig_len = len(pairs)
        self.pairs = [p for p in pairs if p[0] in mapping and p[1] in mapping]
        if (new_len := len(self.pairs)) < orig_len:
            logging.warning(
                f"{orig_len} to {new_len} pairs due to missing embeddings"
            )
        self.unique_keys = set(p[0] for p in self.pairs) | set(p[1] for p in self.pairs)
        # Retain only the elements that are present in the mapping
        self.mapping: Mapping[Any, torch.Tensor] = {
            k: torch.from_numpy(np.array(mapping[k])) for k in self.unique_keys
        }
        self.strategy = strategy
        self.rng = np.random.default_rng(6489 + 6489 + 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, index
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        a, b, label = self.pairs[index]

        if self.strategy == "concat":
            v = torch.concatenate([self.mapping[a], self.mapping[b]], dim=0)
        elif self.strategy == "diff":
            v = self.mapping[a] - self.mapping[b]
        elif self.strategy == "sum":
            v = self.mapping[a] + self.mapping[b]
        elif self.strategy == "separate":
            return (
                self.mapping[a],
                self.mapping[b],
                torch.tensor(label, dtype=torch.float32),
            )

        return (
            v.type(torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class MultiH5:
    def __init__(self, h5_files: List[str] | List[Path]):
        logging.info(f"Loading {len(h5_files)} h5 files: {h5_files}")
        self.h5_files = h5_files
        self.h5s = [h5py.File(f, "r") for f in h5_files]
        self.mapping = ChainMap(*self.h5s)

    def __getitem__(self, key):
        return self.mapping[key]

    def get(self, key, default=None):
        return self.mapping.get(key, default)

    def __contains__(self, key):
        return key in self.mapping

    def __len__(self):
        return sum(len(h5) for h5 in self.h5s)

    def keys(self):
        return self.mapping.keys()

    def values(self):
        return self.mapping.values()

    def items(self):
        return self.mapping.items()

    def __iter__(self):
        for h5 in self.h5s:
            for key in h5:
                yield key

    def close(self):
        for h5 in self.h5s:
            h5.close()


def array_unique_rows(arr: np.ndarray) -> np.ndarray:
    """Get the unique rows of an array."""
    retval = []
    for row in tqdm(arr):
        if retval and np.isclose(row, retval).all(axis=1).any():
            continue
        retval.append(row)
    return np.array(retval)


def random_split(x: SupportsIndex | int, proportions: List[float], seed: int = 6489):
    """Create a random data split.

    If x is an integer, return indices in range(x) to create a random split.
    Otherwise, return subsets of x that corresponding to that random split.
    """
    if isinstance(x, int):
        x = np.arange(x)
    indices = np.arange(len(x))
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices)

    assert np.isclose(sum(proportions), 1), "Proportions must sum to 1"
    p = np.cumsum(proportions)[:-1]  # Points to split
    splits = np.split(indices, np.round(p * len(x), 0).astype(int))
    return [[x[i] for i in idx] for idx in splits]


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    x = rng.random((2000, 10))
    y = rng.random((500, 10))
    arr = np.vstack([x, x, y])

    u = array_unique_rows(arr)
    print(len(u))
