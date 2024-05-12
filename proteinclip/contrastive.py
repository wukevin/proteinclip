import json
from typing import *
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
import onnxruntime

import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from proteinclip.attn import SingleTokenAttention


class UnitNorm(nn.Module):
    """Simple module to normalize the output of a layer to unit norm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


class MLP(nn.Module):
    """Multi-layer perceptron with option to normalize output to unit norm."""

    def __init__(
        self,
        init_dim: int,
        dims: List[int],
        unit_norm: bool = False,
        activation: Callable = nn.GELU,
    ):
        super().__init__()
        self.unit_norm = UnitNorm() if unit_norm else None
        self.dims = dims.copy()
        self.input_shape = (init_dim,)
        self.dims.insert(0, init_dim)
        self.activation = activation

        layers = []
        dim_pairs = list(zip(self.dims[:-1], self.dims[1:]))
        for i, (x, y) in enumerate(dim_pairs):
            layers.append(nn.Linear(x, y))
            if i < len(dim_pairs) - 1:
                layers.append(self.activation())
                layers.append(nn.LayerNorm(y))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retval = self.layers(x)
        if self.unit_norm is not None:
            retval = self.unit_norm(retval)
        return retval


class MLPForRegression(MLP, pl.LightningModule):
    """Baseline wrapper for model around regression task."""

    def __init__(
        self,
        init_dim: int,
        dims: List[int],
        unit_norm: bool = False,
        activation: Callable = nn.GELU,
        lr: float = 1e-4,
    ):
        super().__init__(
            init_dim=init_dim, dims=dims, unit_norm=unit_norm, activation=activation
        )
        self.lr = lr
        self.loss_fn = F.smooth_l1_loss

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise TypeError(f"Unknown batch type: {type(batch)}")

        out = self.forward(x)
        loss = self.loss_fn(out.squeeze(), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise TypeError(f"Unknown batch type: {type(batch)}")

        out = self.forward(x)
        loss = self.loss_fn(out.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True)

        out_np = out.squeeze().cpu().numpy()

        self.log("val_spearman", spearmanr(out_np, y.cpu().numpy())[0], prog_bar=True)
        self.log("val_pearson", pearsonr(out_np, y.cpu().numpy())[0], prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-10)
        return opt


class MLPForClassification(MLP, pl.LightningModule):
    def __init__(
        self,
        init_dim: int,
        dims: List[int],
        unit_norm: bool = False,
        activation: Callable = nn.GELU,
        lr: float = 1e-4,
    ):
        super().__init__(
            init_dim=init_dim, dims=dims, unit_norm=unit_norm, activation=activation
        )
        self.lr = lr
        self.loss_fn = F.binary_cross_entropy_with_logits

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise TypeError(f"Unknown batch type: {type(batch)}")

        out = self.forward(x)
        loss = self.loss_fn(out.squeeze(), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise TypeError(f"Unknown batch type: {type(batch)}")

        out = self.forward(x)
        loss = self.loss_fn(out.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True)

        auroc = roc_auc_score(y.cpu().numpy(), out.squeeze().cpu().numpy())
        self.log("val_auroc", auroc, prog_bar=True)
        auprc = average_precision_score(y.cpu().numpy(), out.squeeze().cpu().numpy())
        self.log("val_auprc", auprc, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-10)
        return opt


class ContrastiveEmbedding(pl.LightningModule):
    """Module for taking embeddings and applying a contrastive loss."""

    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        shared_dim: int,
        num_hidden: int = 1,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.shared_dim = shared_dim
        self.lr = lr

        # Project each input to the shared dim
        # self.project_1 = nn.Sequential(nn.Linear(input_dim_1, shared_dim), UnitNorm())
        # self.project_2 = nn.Sequential(nn.Linear(input_dim_2, shared_dim), UnitNorm())
        self.project_1 = MLP(
            input_dim_1, [input_dim_1] * num_hidden + [shared_dim], unit_norm=True
        )
        self.project_2 = MLP(
            input_dim_2, [input_dim_2] * num_hidden + [shared_dim], unit_norm=True
        )
        self.t = nn.Parameter(data=torch.Tensor([1.0]), requires_grad=True)

    def write_config_json(self, path: str) -> None:
        """Write the configuration to a json file."""
        with open(path, "w") as sink:
            json.dump(
                {
                    "input_dim_1": self.input_dim_1,
                    "input_dim_2": self.input_dim_2,
                    "shared_dim": self.shared_dim,
                    "lr": self.lr,
                },
                sink,
                indent=4,
            )

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
        """Project the two inputs into the shared embedding space and compute
        the contrastive loss between them.
        """
        x1_proj = self.project_1(x_1)
        x2_proj = self.project_2(x_2)

        logits = x1_proj @ x2_proj.T * torch.exp(self.t.to(x1_proj.device))
        labels = torch.arange(x1_proj.shape[0]).to(logits.device)
        l_1 = F.cross_entropy(logits, labels)
        l_2 = F.cross_entropy(logits.T, labels)
        return (l_1 + l_2) / 2

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            x_1, x_2 = batch
        if isinstance(batch, dict):
            x_1, x_2 = batch["x_1"], batch["x_2"]

        loss = self.forward(x_1, x_2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            x_1, x_2 = batch
        if isinstance(batch, dict):
            x_1, x_2 = batch["x_1"], batch["x_2"]

        loss = self.forward(x_1, x_2)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-10)
        return opt

    def query(
        self,
        reference_set: Dict[Any, np.ndarray],
        *,
        query_1: np.ndarray | torch.Tensor | None = None,
        query_2: np.ndarray | torch.Tensor | None = None,
    ) -> List[Tuple[Any, float]]:
        """Query the model with a set of reference embeddings and a query embedding."""
        # depending on whether query1 or query2 is provided, project the other
        # and compute the similarity
        if query_1 is not None:
            assert len(query_1.shape) == 1
            if not isinstance(query_1, torch.Tensor):
                query = torch.from_numpy(query_1).type(torch.float32)
            else:
                query = query_1

            with torch.no_grad():
                query_proj = self.project_1(query)
                ref_proj = self.project_2(
                    torch.tensor(
                        np.array(list(reference_set.values())), dtype=torch.float32
                    )
                )
        elif query_2 is not None:
            assert len(query_2.shape) == 1
            if not isinstance(query_2, torch.Tensor):
                query = torch.from_numpy(query_2).type(torch.float32)
            else:
                query = query_2
            with torch.no_grad():
                query_proj = self.project_2(query)
                ref_proj = self.project_1(
                    torch.tensor(
                        np.array(list(reference_set.values())), dtype=torch.float32
                    )
                )
        else:
            raise ValueError("Either query_1 or query_2 must be provided.")

        # Compute similarities
        sim = (query_proj @ ref_proj.T).cpu().numpy()
        assert sim.size == len(reference_set), f"{sim.size} != {len(reference_set)}"
        return sorted(
            [(k, v) for k, v in zip(reference_set.keys(), sim)], key=lambda x: -x[1]
        )


class ContrastiveEmbeddingWithPreprocessor(ContrastiveEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = SingleTokenAttention(
            embed_dim=self.input_dim_1, num_heads=4
        )

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
        x_1 = self.preprocessor(x_1)
        return super().forward(x_1, x_2)


class ContrastiveEmbeddingOneToMany(ContrastiveEmbedding):
    """Compared to standard contrastive learning context, assume that pairing
    is many-to-one, where the FIRST item in the pair is the item that may be
    repeated, e.g., (foo, bar), (foo, baz), (foo, qux), etc.)"""

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
        """Project both inputs to shared embedding space and compute loss."""
        x_1_norm = x_1 / torch.linalg.vector_norm(x_1, dim=-1, keepdim=True)
        x_1_sim = x_1_norm @ x_1_norm.T

        # NxN matrix if the ith element is the same as the jth element
        # i.e., an indicator matrix that 1 if [i, j] are related and 0 otherwise
        shared_x_1 = torch.isclose(x_1_sim, torch.ones_like(x_1_sim)).bool()

        x1_proj = self.project_1(x_1)
        x2_proj = self.project_2(x_2)

        logits = x1_proj @ x2_proj.T * torch.exp(self.t.to(x1_proj.device))

        # Push similar things closer, and things that are different farther

        loss1 = -torch.log(
            (shared_x_1 * torch.exp(logits)).sum(dim=1) / torch.exp(logits).sum(dim=1)
        ).mean()
        loss2 = -torch.log(
            (shared_x_1 * torch.exp(logits)).sum(dim=0) / torch.exp(logits).sum(dim=0)
        ).mean()
        return (loss1 + loss2) / 2


class ContrastiveEmbeddingRegression(pl.LightningModule):
    """
    A 'contrastive' network that does regression. Originally described in Singh
    et al. in https://www.pnas.org/doi/10.1073/pnas.2220778120. The model takes
    two embeddings of dimension h and j, and learns the following network:
    - Nonlinear projection to shared embedding e of dim d: e = ReLU(Wx + b)
    - Take the dot product of the two embeddings: z = e1 @ e2
    - Do a ReLU(z) to get the final regression score
    """

    def __init__(
        self,
        dim_1: int,
        dim_2: int,
        shared_dim: int = 512,
        final_relu: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.project_1 = nn.Linear(dim_1, shared_dim)
        self.project_2 = nn.Linear(dim_2, shared_dim)
        self.final_relu = final_relu
        self.lr = lr

    def forward(
        self, x_1: torch.Tensor, x_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        e1 = F.relu(self.project_1(x_1))
        e2 = F.relu(self.project_2(x_2))
        score = (e1 * e2).sum(axis=-1)
        if self.final_relu:
            score = F.relu(score)
        return score.squeeze()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x_1, x_2, y = batch
        if isinstance(batch, dict):
            x_1, x_2, y = batch["x_1"], batch["x_2"], batch["y"]
        y = y.squeeze()

        scores = self.forward(x_1, x_2)
        loss = F.mse_loss(scores, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x_1, x_2, y = batch
        if isinstance(batch, dict):
            x_1, x_2, y = batch["x_1"], batch["x_2"], batch["y"]
        y = y.squeeze()

        scores = self.forward(x_1, x_2)
        loss = F.smooth_l1_loss(scores, y)
        self.log("val_loss", loss, prog_bar=True)

        self.log(
            "val_spearman",
            spearmanr(scores.cpu().numpy(), y.cpu().numpy())[0],
            prog_bar=True,
        )
        self.log(
            "val_pearson",
            pearsonr(scores.cpu().numpy(), y.cpu().numpy())[0],
            prog_bar=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-10)
        return opt


def model_to_onnx(
    model: nn.Module, path: str, input_shape: Sequence[int] | None = None
) -> None:
    """Convert a PyTorch model to ONNX format and check it.

    Input shape is expected WITHOUT initial batch dimension."""
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    if input_shape is None:
        input_shape = model.input_shape
    x = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        x,
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    # Load the ONNX version and check that the inputs produce the same outputs.
    ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    to_numpy = lambda t: (
        t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()
    )
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare the outputs
    with torch.no_grad():
        torch_out = model(x)
    assert np.allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == "__main__":
    # Toy run
    b, n, e = 2, 4, 4
    model = TransformerAggregation(embed_dim=e, num_heads=2).eval().cuda()
    dev = next(model.parameters()).device

    rng = np.random.default_rng(1234)
    x = rng.random((b, n, e))
    # print(x)
    x = torch.from_numpy(x).to(dev).type(torch.float32)

    with torch.no_grad():
        y = model(x)
        y2 = model(torch.concat([x[1].unsqueeze(0), x[0].unsqueeze(0)]))
    print(y)
    print(y2)
