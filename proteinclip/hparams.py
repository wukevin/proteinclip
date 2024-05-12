import json
from pathlib import Path
from typing import Any, Callable, Literal

import torch


class HParams:
    """Hyperparameter container."""

    def __init__(
        self,
        batch_size: int = 256,
        max_epochs: int = 100,
        learning_rate: float = 1e-4,
        eps: float = 1e-10,
        optimizer: Literal["adamw", "adam"] = "adamw",
    ):
        # Copies from class variables into instance variables
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.eps = eps
        self.optimizer = optimizer

    def __str__(self) -> str:
        return f"HParams[{str(self.as_dict())}]"

    def as_dict(self) -> dict:
        # Subset to non-private and non-function attributes
        d = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        return d

    def to_json(self, fname: str | Path) -> None:
        with open(fname, "w") as sink:
            json.dump(self.as_dict(), sink, indent=2)

    def make_optim_factory(self) -> Callable[[Any], torch.optim.Optimizer]:
        if self.optimizer == "adamw":
            pfunc = lambda x: torch.optim.AdamW(x, lr=self.learning_rate, eps=self.eps)
        elif self.optimizer == "adam":
            pfunc = lambda x: torch.optim.Adam(x, lr=self.learning_rate, eps=self.eps)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        return pfunc


def read_hparams(fname: str | Path) -> HParams:
    """Read hyperparameters from a .json file."""
    if not fname:
        return HParams()
    with open(fname) as source:
        return HParams(**json.load(source))


if __name__ == "__main__":
    x = read_hparams("/home/wukevin/projects/gpt-protein/configs/hparams.json")
    print(x)
    print(x.make_optim_factory())
    print(x.as_dict())
    print(x.batch_size, x.max_epochs)
