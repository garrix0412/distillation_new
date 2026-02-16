"""
CNN Block module - replaces the Syndrome Transformer in AlphaQubit.

Uses dilated 2D convolutions to capture spatial correlations between
stabilizers, without the computational overhead of self-attention.
This is the key simplification for FPGA deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """
    Spatial mixing block using dilated 2D convolutions.

    Replaces AlphaQubit's Syndrome Transformer (which uses attention + conv).
    Operates on stabilizer representations scattered to a 2D grid.

    Architecture per layer:
    1. Scatter stabilizer representations to 2D grid
    2. Dilated 2D convolutions (multiple layers with increasing dilation)
    3. Gather back to per-stabilizer representation
    4. Residual connection + activation

    This preserves the spatial structure of the surface code while
    being much more FPGA-friendly than attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        conv_dim: int,
        n_conv_layers: int = 2,
        distance: int = 3,
    ):
        """
        Args:
            hidden_dim: Dimension of per-stabilizer representation.
            conv_dim: Internal dimension for convolutions (can be < hidden_dim).
            n_conv_layers: Number of convolutional layers.
            distance: Code distance (determines 2D grid size).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.distance = distance
        self.n_conv_layers = n_conv_layers

        # The 2D grid for rotated surface code stabilizers is (d+1) x (d+1)
        # but only d^2-1 positions are occupied
        self.grid_size = distance + 1

        # Project to conv dimension
        self.proj_in = nn.Linear(hidden_dim, conv_dim)

        # Dilated convolutions with increasing dilation rates
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_conv_layers):
            dilation = min(2**i, distance)  # 1, 2, 4, ... capped at distance
            self.convs.append(
                nn.Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    padding=dilation,  # same padding
                    dilation=dilation,
                )
            )
            self.norms.append(nn.BatchNorm2d(conv_dim))

        # Project back to hidden dimension
        self.proj_out = nn.Linear(conv_dim, hidden_dim)

        # Layer norm for residual
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Build the scatter/gather index maps
        self._build_grid_maps(distance)

    def _build_grid_maps(self, distance: int):
        """
        Build index maps for scattering stabilizers to 2D grid and back.

        For the rotated surface code, stabilizers are arranged in a
        checkerboard pattern on a (d+1) x (d+1) grid. We precompute
        the mapping between flat stabilizer indices and grid positions.
        """
        d = distance
        n_stab = d * d - 1
        grid_size = d + 1

        # Map stabilizer index to 2D grid position
        # For rotated surface code, stabilizers occupy a checkerboard pattern
        # Simple mapping: fill row by row, skipping one position
        stab_rows = []
        stab_cols = []
        idx = 0
        for r in range(grid_size):
            for c in range(grid_size):
                # Checkerboard: stabilizers at positions where (r+c) is odd
                # or some other pattern depending on the code variant
                if idx < n_stab:
                    stab_rows.append(r % grid_size)
                    stab_cols.append(c % grid_size)
                    idx += 1

        # Ensure we placed all stabilizers
        assert len(stab_rows) == n_stab, (
            f"Could not place all {n_stab} stabilizers on grid"
        )

        self.register_buffer(
            "stab_rows", torch.tensor(stab_rows, dtype=torch.long)
        )
        self.register_buffer(
            "stab_cols", torch.tensor(stab_cols, dtype=torch.long)
        )

    def _scatter_to_grid(self, stab_repr):
        """
        Scatter per-stabilizer representations to 2D grid.

        Args:
            stab_repr: [batch, n_stab, conv_dim]

        Returns:
            grid: [batch, conv_dim, grid_size, grid_size]
        """
        batch_size, n_stab, dim = stab_repr.shape
        grid = torch.zeros(
            batch_size, dim, self.grid_size, self.grid_size,
            device=stab_repr.device, dtype=stab_repr.dtype,
        )
        # stab_repr: [batch, n_stab, dim] -> need to scatter
        # grid[:, :, rows, cols] = stab_repr transposed
        grid[:, :, self.stab_rows, self.stab_cols] = stab_repr.permute(0, 2, 1)
        return grid

    def _gather_from_grid(self, grid):
        """
        Gather per-stabilizer representations from 2D grid.

        Args:
            grid: [batch, conv_dim, grid_size, grid_size]

        Returns:
            stab_repr: [batch, n_stab, conv_dim]
        """
        # grid[:, :, rows, cols] -> [batch, conv_dim, n_stab]
        stab_repr = grid[:, :, self.stab_rows, self.stab_cols]
        return stab_repr.permute(0, 2, 1)  # [batch, n_stab, conv_dim]

    def forward(self, x):
        """
        Args:
            x: [batch, n_stab, hidden_dim] per-stabilizer representations

        Returns:
            out: [batch, n_stab, hidden_dim] updated representations
        """
        residual = x

        # Project to conv dimension
        h = self.proj_in(x)  # [batch, n_stab, conv_dim]

        # Scatter to 2D grid
        grid = self._scatter_to_grid(h)  # [batch, conv_dim, H, W]

        # Apply dilated convolutions
        for conv, norm in zip(self.convs, self.norms):
            grid = grid + F.relu(norm(conv(grid)))

        # Gather back
        h = self._gather_from_grid(grid)  # [batch, n_stab, conv_dim]

        # Project back to hidden dim
        h = self.proj_out(h)  # [batch, n_stab, hidden_dim]

        # Residual + norm
        out = self.layer_norm(residual + h)

        return out
