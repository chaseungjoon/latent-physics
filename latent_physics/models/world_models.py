"""World models. All map inputs (B, T, D_in) -> standardized deltas (B, T, S)."""
from __future__ import annotations

import torch
from torch import nn


class GRUWorldModel(nn.Module):
    """Recurrent world model: consumes (s_t, a_t) and predicts s_{t+1} - s_t.

    The recurrent carry is the only place trajectory-level information (mass, friction, ...) can
    accumulate, so the per-layer GRU states are the main object of the representation analysis.
    Layers are separate single-layer GRUs so every layer's state sequence is observable.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.grus = nn.ModuleList(nn.GRU(hidden, hidden, batch_first=True) for _ in range(layers))
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    @property
    def gru_names(self) -> list[str]:
        return [f"gru{i + 1}" for i in range(len(self.grus))]

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        z = self.enc(x)
        hidden = {"enc": z}
        for name, gru in zip(self.gru_names, self.grus):
            z, _ = gru(z)
            hidden[name] = z
        y = self.dec(z)
        return (y, hidden) if return_hidden else y

    def step(self, x_t: torch.Tensor, carry: list[torch.Tensor] | None = None, edit=None):
        """One timestep. x_t: (B, D); carry: per-GRU-layer (B, H) states or None (zeros).

        `edit(layer_idx, h) -> h` is applied to each GRU layer's output, which is also its next carry,
        so it can be used for ablations that persist through time.
        """
        z = self.enc(x_t)
        new_carry = []
        for i, gru in enumerate(self.grus):
            h0 = None if carry is None else carry[i].unsqueeze(0).contiguous()
            out, _ = gru(z.unsqueeze(1), h0)
            z = out[:, 0]
            if edit is not None:
                z = edit(i, z)
            new_carry.append(z)
        return self.dec(z), new_carry

    def forward_stepwise(self, x: torch.Tensor, edit=None) -> torch.Tensor:
        carry, ys = None, []
        for t in range(x.shape[1]):
            y, carry = self.step(x[:, t], carry, edit)
            ys.append(y)
        return torch.stack(ys, 1)


class MLPWorldModel(nn.Module):
    """Memoryless model (s_t, a_t[, phi]) -> delta, applied independently at every timestep.

    Without phi it cannot infer hidden parameters and sets the "physics unknown" error floor;
    with the true phi appended (oracle) it sets the "physics known" ceiling.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.net = nn.Sequential(*mods, nn.Linear(d, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x_t: torch.Tensor, carry=None, edit=None):
        return self.net(x_t), None
