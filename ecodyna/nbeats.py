from typing import Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NBEATS(nn.Module):
    """
    Heavily inspired from
    https://github.com/unit8co/darts/blob/e3f33094ba0b02a96b25db9101b2be6def2d9fc7/darts/models/forecasting/nbeats.py
    """
    class _Block(nn.Module):
        def __init__(
                self,
                n_in: int,
                n_out: int,
                n_layers: int,
                expansion_coefficient_dim: int,
                layer_width: int
        ):
            super().__init__()
            self.n_in = n_in
            self.n_out = n_out
            self.n_layers = n_layers
            self.expansion_coefficient_dim = expansion_coefficient_dim
            self.layer_widths = layer_width

            self.FC_stack = nn.ModuleList(
                [nn.Linear(n_in, layer_width)] + [nn.Linear(layer_width, layer_width) for _ in range(n_layers - 1)]
            )
            self.FC_backcast = nn.Linear(layer_width, expansion_coefficient_dim)
            self.FC_forecast = nn.Linear(layer_width, expansion_coefficient_dim)

            self.g_backcast = nn.Linear(expansion_coefficient_dim, n_in)
            self.g_forecast = nn.Linear(expansion_coefficient_dim, n_out)

        def forward(self, x: Tensor):
            for layer in self.FC_stack:
                x = F.relu(layer(x))

            backcast = self.g_backcast(self.FC_backcast(x))
            forecast = self.g_forecast(self.FC_forecast(x))
            return backcast, forecast

    class _Stack(nn.Module):
        def __init__(self, n_in: int, n_out: int, n_blocks: int, *args, **kwargs):
            super().__init__()
            self.n_in = n_in
            self.n_out = n_out
            self.n_blocks = n_blocks

            self.blocks = nn.ModuleList([NBEATS._Block(n_in, n_out, *args, **kwargs) for _ in range(n_blocks)])

        def forward(self, x: Tensor):
            B, T = x.size()
            assert T == self.n_in, f'NBeats Stack should take {self.n_in} time steps as input'
            forecast = torch.zeros(B, self.n_out)
            for block in self.blocks:
                block_backcast, block_forecast = block(x)
                x = x - block_backcast
                forecast += block_forecast
            return x, forecast

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_stacks: int,
            n_blocks: int,
            n_layers: int,
            expansion_coefficient_dim: int,
            layer_widths: Union[int, List[int]]
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.layer_widths = layer_widths
        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * n_stacks

        self.stacks = nn.ModuleList([
            NBEATS._Stack(
                n_in=n_in,
                n_out=n_out,
                n_blocks=n_blocks,
                n_layers=n_layers,
                layer_width=self.layer_widths[i],
                expansion_coefficient_dim=expansion_coefficient_dim
            )
            for i in range(n_stacks)
        ])
        self.stacks[-1].blocks[-1].FC_backcast.requires_grad_(False)
        self.stacks[-1].blocks[-1].g_backcast.requires_grad_(False)

    def forward(self, x: Tensor):
        B, T = x.size()
        assert T == self.n_in, f'NBeats should take {self.n_in} time steps as input'
        backcast = x
        forecast = torch.zeros(B, self.n_out)
        for stack in self.stacks:
            backcast, stack_forecast = stack(backcast)
            forecast += stack_forecast
        return forecast
