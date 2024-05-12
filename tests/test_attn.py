import unittest

import torch
import numpy as np

from proteinclip import attn


class TestSingleTokenAttention(unittest.TestCase):

    def test_permutation_on_batch(self):
        """Test that permutation on batch axis doesn't affect result."""
        b, n, e = 16, 8, 4
        dev = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        rng = np.random.default_rng(1234)
        x = rng.random((b, n, e))

        net = attn.SingleTokenAttention(embed_dim=e).to(dev)
        y = net.forward(torch.from_numpy(x).type(torch.float32).to(dev))
        self.assertEqual(y.shape, (b, e))

        shuffle_idx = np.arange(b)
        rng.shuffle(shuffle_idx)
        x_shuf = x[shuffle_idx]
        y_shuf = net.forward(torch.from_numpy(x_shuf).type(torch.float32).to(dev))
        self.assertTrue(torch.allclose(y[shuffle_idx], y_shuf))

    def test_permutation_on_batch_with_padding(self):
        """Test that permutation on batch axis doesn't affect results with padding."""
        b, n, e = 8, 128, 32
        dev = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        rng = np.random.default_rng(1234)
        x = rng.random((b, n, e))
        lengths = rng.integers(64, 80, size=b)  # Add padding
        for i, l in enumerate(lengths):
            x[i, l:] = 0.0

        net = attn.SingleTokenAttention(embed_dim=e).to(dev)
        y = net.forward(torch.from_numpy(x).type(torch.float32).to(dev))
        self.assertEqual(y.shape, (b, e))

        shuffle_idx = np.arange(b)
        rng.shuffle(shuffle_idx)
        x_shuf = x[shuffle_idx]
        y_shuf = net.forward(torch.from_numpy(x_shuf).type(torch.float32).to(dev))
        self.assertTrue(torch.allclose(y[shuffle_idx], y_shuf))


if __name__ == "__main__":
    unittest.main()
