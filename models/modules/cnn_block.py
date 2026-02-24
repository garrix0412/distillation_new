"""
CNN Block 模块 - 替代 AlphaQubit 中的 Syndrome Transformer。

使用空洞 2D 卷积捕获 stabilizer 之间的空间相关性，
避免自注意力机制带来的计算开销。
这是面向 FPGA 部署的核心简化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """
    基于空洞 2D 卷积的空间混合模块。

    替代 AlphaQubit 的 Syndrome Transformer（使用注意力 + 卷积）。
    对散布到 2D 网格上的 stabilizer 表征进行操作。

    每层架构：
    1. 将 stabilizer 表征散布到 2D 网格
    2. 空洞 2D 卷积（多层，膨胀率递增）
    3. 从网格收集回 per-stabilizer 表征
    4. 残差连接 + 激活

    保留了 surface code 的空间结构，
    同时比注意力机制更适合 FPGA 实现。
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
            hidden_dim: per-stabilizer 表征的维度。
            conv_dim: 卷积内部维度（可小于 hidden_dim）。
            n_conv_layers: 卷积层数。
            distance: 码距（决定 2D 网格大小）。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.distance = distance
        self.n_conv_layers = n_conv_layers

        # 2D 网格大小将由 _build_grid_maps 根据
        # 实际 stim 检测器坐标确定
        # （通常为 (d+1) x (d+1)，但动态确定）

        # 投影到卷积维度
        self.proj_in = nn.Linear(hidden_dim, conv_dim)

        # 膨胀率递增的空洞卷积
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_conv_layers):
            dilation = min(2**i, distance)  # 1, 2, 4, ... 上限为码距
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

        # 投影回隐藏维度
        self.proj_out = nn.Linear(conv_dim, hidden_dim)

        # 残差连接用的 LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 构建 scatter/gather 索引映射
        self._build_grid_maps(distance)

    def _build_grid_maps(self, distance: int):
        """
        构建将 stabilizer 散布到 2D 网格及反向收集的索引映射。

        使用 stim 从 rotated surface code 电路中提取实际检测器坐标，
        确保网格映射与物理 stabilizer 布局一致（棋盘格/菱形排列）。

        使用中间轮坐标，因为边界轮可能由于 stim 的边界处理
        而具有不同的检测器排序。
        """
        import stim

        d = distance
        n_stab = d * d - 1

        # 生成最小电路以提取检测器坐标
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=max(3, d),
            after_clifford_depolarization=0.001,
            after_reset_flip_probability=0.001,
            before_measure_flip_probability=0.001,
            before_round_data_depolarization=0.001,
        )
        coords = circuit.get_detector_coordinates()

        # 使用中间轮检测器（扁平索引 n_stab .. 2*n_stab-1），
        # 对应一个完整的内部轮，具有一致的排序
        xs, ys = set(), set()
        for i in range(n_stab):
            det_idx = n_stab + i
            xs.add(coords[det_idx][0])
            ys.add(coords[det_idx][1])

        x_to_col = {x: i for i, x in enumerate(sorted(xs))}
        y_to_row = {y: i for i, y in enumerate(sorted(ys))}

        stab_rows = []
        stab_cols = []
        for i in range(n_stab):
            det_idx = n_stab + i
            stab_rows.append(y_to_row[coords[det_idx][1]])
            stab_cols.append(x_to_col[coords[det_idx][0]])

        self.grid_size = max(len(ys), len(xs))

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
        将 per-stabilizer 表征散布到 2D 网格。

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
        # stab_repr: [batch, n_stab, dim] -> 需要散布
        # grid[:, :, rows, cols] = stab_repr 转置后
        grid[:, :, self.stab_rows, self.stab_cols] = stab_repr.permute(0, 2, 1)
        return grid

    def _gather_from_grid(self, grid):
        """
        从 2D 网格收集 per-stabilizer 表征。

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
            x: [batch, n_stab, hidden_dim] per-stabilizer 表征

        Returns:
            out: [batch, n_stab, hidden_dim] 更新后的表征
        """
        residual = x

        # 投影到卷积维度
        h = self.proj_in(x)  # [batch, n_stab, conv_dim]

        # 散布到 2D 网格
        grid = self._scatter_to_grid(h)  # [batch, conv_dim, H, W]

        # 应用空洞卷积
        for conv, norm in zip(self.convs, self.norms):
            grid = grid + F.relu(norm(conv(grid)))

        # 收集回来
        h = self._gather_from_grid(grid)  # [batch, n_stab, conv_dim]

        # 投影回隐藏维度
        h = self.proj_out(h)  # [batch, n_stab, hidden_dim]

        # 残差 + 归一化
        out = self.layer_norm(residual + h)

        return out
