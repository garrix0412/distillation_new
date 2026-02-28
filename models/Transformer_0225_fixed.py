import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import stim
import argparse
import numpy as np
import math
import time
import os
from dataclasses import dataclass
import contextlib

# --- 基础设施 ---
def setup_ddp():
    if "RANK" not in os.environ: return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

# ==============================================================================
# 物理映射层: 联合映射 X + Z stabilizer
# ==============================================================================
@dataclass
class FullMappingInfo:
    # Z stabilizer 映射
    gather_z: torch.Tensor          # [num_t * num_z] → stim detector index
    valid_z: torch.Tensor           # [num_t * num_z]
    z_neighbors: torch.Tensor       # [num_z, 4] → 4个对角Z邻居在z_locs中的index
    z_hint_neighbors: torch.Tensor  # [num_t * num_z, 4] → 4个最近X邻居的stim index
    num_z: int
    # X stabilizer 映射
    gather_x: torch.Tensor          # [num_t * num_x] → stim detector index
    valid_x: torch.Tensor           # [num_t * num_x]
    x_neighbors: torch.Tensor       # [num_x, 4] → 4个对角X邻居在x_locs中的index
    x_hint_neighbors: torch.Tensor  # [num_t * num_x, 4] → 4个最近Z邻居的stim index
    num_x: int
    # 共有
    num_t: int
    rounds: int


class FullMapper(nn.Module):
    """联合映射 X 和 Z stabilizer，取代原来的 StrictZMapper"""
    
    def __init__(self, d: int, rounds: int):
        super().__init__()
        self.d = d
        self.rounds = rounds
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds
        )
        self.mapping_info = self._build_full_mapping()
        
        # 注册所有 buffer
        self.register_buffer('gather_z', self.mapping_info.gather_z)
        self.register_buffer('valid_z', self.mapping_info.valid_z)
        self.register_buffer('z_neighbors', self.mapping_info.z_neighbors)
        self.register_buffer('z_hint_neighbors', self.mapping_info.z_hint_neighbors)
        self.register_buffer('gather_x', self.mapping_info.gather_x)
        self.register_buffer('valid_x', self.mapping_info.valid_x)
        self.register_buffer('x_neighbors', self.mapping_info.x_neighbors)
        self.register_buffer('x_hint_neighbors', self.mapping_info.x_hint_neighbors)

    def _build_full_mapping(self) -> FullMappingInfo:
        coords = self.circuit.get_detector_coordinates()
        if not coords:
            empty_long = torch.empty(0, dtype=torch.long)
            empty_float = torch.empty(0, dtype=torch.float32)
            return FullMappingInfo(
                empty_long, empty_float, empty_long.view(0, 4), empty_long.view(0, 4), 0,
                empty_long, empty_float, empty_long.view(0, 4), empty_long.view(0, 4), 0,
                0, self.rounds
            )
        
        # 1. 分离 Z 和 X 的空间坐标
        # 在 rotated_memory_z 中，最后一轮的 detector 都是 Z type
        max_t = max(t for _, _, t in coords.values())
        
        all_locs_with_type = {}  # (r,c) → 'Z' or 'X'
        z_locs_set = set()
        
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            if math.isclose(t, max_t):
                z_locs_set.add((r, c))
        
        x_locs_set = set()
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            if (r, c) not in z_locs_set:
                x_locs_set.add((r, c))
        
        z_locs = sorted(list(z_locs_set), key=lambda p: (p[0], p[1]))
        x_locs = sorted(list(x_locs_set), key=lambda p: (p[0], p[1]))
        
        loc_to_idx_z = {loc: i for i, loc in enumerate(z_locs)}
        loc_to_idx_x = {loc: i for i, loc in enumerate(x_locs)}
        
        num_z = len(z_locs)
        num_x = len(x_locs)
        
        # 2. 收集所有时间步
        unique_times = sorted(list(set(t for _, _, t in coords.values())))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        num_t = len(unique_times)
        
        # 建立反查字典: (r, c, t) → stim detector index
        coord_time_to_idx = {}
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            coord_time_to_idx[(r, c, t)] = idx
        
        # 3. 构建 Z 的 gather 矩阵和 valid mask
        gather_z = torch.zeros((num_t, num_z), dtype=torch.long)
        valid_z = torch.zeros((num_t, num_z), dtype=torch.float32)
        
        for t_idx, t in enumerate(unique_times):
            for z_idx, (zr, zc) in enumerate(z_locs):
                if (zr, zc, t) in coord_time_to_idx:
                    gather_z[t_idx, z_idx] = coord_time_to_idx[(zr, zc, t)]
                    valid_z[t_idx, z_idx] = 1.0
        
        # 4. 构建 X 的 gather 矩阵和 valid mask
        gather_x = torch.zeros((num_t, num_x), dtype=torch.long)
        valid_x = torch.zeros((num_t, num_x), dtype=torch.float32)
        
        for t_idx, t in enumerate(unique_times):
            for x_idx, (xr, xc) in enumerate(x_locs):
                if (xr, xc, t) in coord_time_to_idx:
                    gather_x[t_idx, x_idx] = coord_time_to_idx[(xr, xc, t)]
                    valid_x[t_idx, x_idx] = 1.0
        
        # 5. Z 的 4 个对角 Z 邻居 (用于空间 embedding)
        z_neighbors = torch.full((num_z, 4), num_z, dtype=torch.long)  # padding = num_z
        for i, (r, c) in enumerate(z_locs):
            for j, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                if (r + dr, c + dc) in loc_to_idx_z:
                    z_neighbors[i, j] = loc_to_idx_z[(r + dr, c + dc)]
        
        # 6. X 的 4 个对角 X 邻居 (用于空间 embedding)
        x_neighbors = torch.full((num_x, 4), num_x, dtype=torch.long)  # padding = num_x
        for i, (r, c) in enumerate(x_locs):
            for j, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                if (r + dr, c + dc) in loc_to_idx_x:
                    x_neighbors[i, j] = loc_to_idx_x[(r + dr, c + dc)]
        
        # 7. Z 的 4 个最近 X 邻居 (hint，按距离排序)
        z_hint_neighbors = torch.zeros((num_t, num_z, 4), dtype=torch.long)
        z_to_nearest_x = {}
        for zr, zc in z_locs:
            sorted_x = sorted(list(x_locs_set), key=lambda p: (p[0]-zr)**2 + (p[1]-zc)**2)
            z_to_nearest_x[(zr, zc)] = sorted_x[:4]
        
        for t_idx, t in enumerate(unique_times):
            for z_idx, (zr, zc) in enumerate(z_locs):
                for i, (xr, xc) in enumerate(z_to_nearest_x[(zr, zc)]):
                    if (xr, xc, t) in coord_time_to_idx:
                        z_hint_neighbors[t_idx, z_idx, i] = coord_time_to_idx[(xr, xc, t)]
        
        # 8. X 的 4 个最近 Z 邻居 (hint，按距离排序)
        x_hint_neighbors = torch.zeros((num_t, num_x, 4), dtype=torch.long)
        x_to_nearest_z = {}
        for xr, xc in x_locs:
            sorted_z = sorted(list(z_locs_set), key=lambda p: (p[0]-xr)**2 + (p[1]-xc)**2)
            x_to_nearest_z[(xr, xc)] = sorted_z[:4]
        
        for t_idx, t in enumerate(unique_times):
            for x_idx, (xr, xc) in enumerate(x_locs):
                for i, (zr, zc) in enumerate(x_to_nearest_z[(xr, xc)]):
                    if (zr, zc, t) in coord_time_to_idx:
                        x_hint_neighbors[t_idx, x_idx, i] = coord_time_to_idx[(zr, zc, t)]
        
        return FullMappingInfo(
            gather_z=gather_z.flatten(),
            valid_z=valid_z.flatten(),
            z_neighbors=z_neighbors,
            z_hint_neighbors=z_hint_neighbors.view(-1, 4),
            num_z=num_z,
            gather_x=gather_x.flatten(),
            valid_x=valid_x.flatten(),
            x_neighbors=x_neighbors,
            x_hint_neighbors=x_hint_neighbors.view(-1, 4),
            num_x=num_x,
            num_t=num_t,
            rounds=self.rounds,
        )

    def get_spatial_coords(self, stab_type='both'):
        """
        获取归一化空间坐标。
        
        注意 rope_base_d 的选择:
        - 固定 BASE_D (如 7.0) 可以保持不同 distance 间相邻 stabilizer 的坐标间距不变，
          有利于从小 d 到大 d 的 transfer learning（RoPE 角度差保持一致）。
          代价是大 d 时坐标会超出 [-1, 1]，但 RoPE 是周期性的，影响较小。
        - 按当前 d 归一化则所有 d 的坐标范围统一，但 transfer 时邻居间的角度变了。
        
        这里默认用固定 BASE_D=7.0 以保持 transfer 兼容性。
        """
        coords = self.circuit.get_detector_coordinates()
        BASE_D = 23.0  # 设为长期最大目标 d，确保所有 d≤23 的坐标在安全范围内
        
        def _get_coords_for_locs(gather_flat, num_spatial):
            indices = gather_flat[:num_spatial].tolist()
            xy = []
            for idx in indices:
                if idx in coords:
                    x, y, _ = coords[idx]
                else:
                    x, y = 0, 0
                nx = x / (2 * BASE_D) * 2 - 1
                ny = y / (2 * BASE_D) * 2 - 1
                xy.append([nx, ny])
            return torch.tensor(xy, dtype=torch.float32)
        
        z_coords = _get_coords_for_locs(self.mapping_info.gather_z, self.mapping_info.num_z)
        x_coords = _get_coords_for_locs(self.mapping_info.gather_x, self.mapping_info.num_x)
        
        if stab_type == 'z': return z_coords
        if stab_type == 'x': return x_coords
        return torch.cat([z_coords, x_coords], dim=0)  # [num_z + num_x, 2]


# ==============================================================================
# 组件
# ==============================================================================

class CoordinateRoPE(nn.Module):
    """
    2D Rotary Position Embedding — 按照 AlphaQubit2 论文:
    "Half of the channels use the x coordinate and half use the y coordinate"
    
    前半 channels 用 x 坐标旋转，后半 channels 用 y 坐标旋转。
    每半内部用标准 rotate_half。
    """
    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
        self.head_dim = head_dim
        self.half_dim = head_dim // 2       # 每个坐标维度占的 channels 数
        self.quarter_dim = head_dim // 4    # 每个坐标的 sin/cos 频率数

        inv_freq = 1.0 / (100 ** (torch.arange(0, self.quarter_dim, dtype=torch.float32) / self.quarter_dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_freqs(self, coords, device):
        """
        返回 (freqs_x, freqs_y)，各自 shape [N, half_dim]
        """
        x = coords[:, 0].to(device)
        y = coords[:, 1].to(device)

        fx = torch.einsum("i,j->ij", x, self.inv_freq)  # [N, quarter_dim]
        fy = torch.einsum("i,j->ij", y, self.inv_freq)  # [N, quarter_dim]

        # 各自重复一份，用于 rotate_half 的 sin/cos 对
        freqs_x = torch.cat([fx, fx], dim=-1)  # [N, half_dim]
        freqs_y = torch.cat([fy, fy], dim=-1)  # [N, half_dim]
        return freqs_x, freqs_y


def apply_rope_2d(q, k, freqs_x, freqs_y):
    """
    对 q, k 应用 2D RoPE。
    q, k: [B, N, n_heads, head_dim]
    freqs_x, freqs_y: [N, half_dim]  (会 broadcast 到 [1, N, 1, half_dim])
    """
    half = q.shape[-1] // 2
    
    # 拆分为 x-half 和 y-half
    q_x, q_y = q[..., :half], q[..., half:]
    k_x, k_y = k[..., :half], k[..., half:]
    
    # broadcast freqs
    fx = freqs_x.unsqueeze(0).unsqueeze(2)  # [1, N, 1, half_dim]
    fy = freqs_y.unsqueeze(0).unsqueeze(2)
    
    # 对每半应用标准 rotate_half
    def _rotate_half(t, freqs):
        t_rot = torch.cat((-t[..., t.shape[-1]//2:], t[..., :t.shape[-1]//2]), dim=-1)
        return t * freqs.cos() + t_rot * freqs.sin()
    
    q_x = _rotate_half(q_x, fx)
    k_x = _rotate_half(k_x, fx)
    q_y = _rotate_half(q_y, fy)
    k_y = _rotate_half(k_y, fy)
    
    return torch.cat([q_x, q_y], dim=-1), torch.cat([k_x, k_y], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * x.to(torch.float32).pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * x


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- 轻量 RNN (per-stabilizer GRU) ---
class RecurrentBlock(nn.Module):
    """AQ2 style lightweight recurrent block: per-stabilizer GRU + residual"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.d_model = d_model
    
    def forward(self, x, h):
        """
        x: [B, N, D] — 当前时间步的 stabilizer 表示
        h: [B, N, D] — 上一时间步的隐状态
        returns: (output [B,N,D], new_hidden [B,N,D])
        """
        B, N, D = x.shape
        x_normed = self.norm(x).reshape(B * N, D)
        h_flat = h.reshape(B * N, D)
        new_h = self.gru(x_normed, h_flat).reshape(B, N, D)
        return x + new_h, new_h  # 残差连接


# --- Spatial Transformer Block (仅空间注意力) ---
class SpatialTransformerBlock(nn.Module):
    """纯空间 self-attention，不再携带时间状态"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim); self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = SwiGLU(dim, 4 * dim)

    def forward(self, x, freqs_x, freqs_y):
        """
        x: [B, N, D]
        freqs_x, freqs_y: [N, half_dim]
        """
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = map(
            lambda t: t.view(B, N, self.n_heads, self.head_dim),
            qkv.chunk(3, dim=-1)
        )
        
        # 2D RoPE
        q, k = apply_rope_2d(q, k, freqs_x, freqs_y)
        
        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        x = x + self.proj(out.transpose(1, 2).reshape(B, N, D))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Cross Attention Readout (不动) ---
class AQCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, q, kv, padding_mask):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, key_padding_mask=padding_mask)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


# ==============================================================================
# 主模型: AlphaQubit V2 — X+Z 联合处理, RNN+TF 交替架构
# ==============================================================================
class AlphaQubitV2(nn.Module):
    """
    架构遵循 AQ2 论文的 RNN-Transformer 交替模式:
        RNN → RNN → [3 TF] → RNN → [3 TF] → RNN → [3 TF] → RNN
    
    共 5 个 RNN + 9 个 Transformer 层。
    X 和 Z stabilizer 拼接为一个序列，共同参与 attention。
    Embedding 和 Readout 保持原有设计不变。
    """
    def __init__(self, mapper, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        
        self.num_z = mapper.mapping_info.num_z
        self.num_x = mapper.mapping_info.num_x
        self.num_stab = self.num_z + self.num_x  # 总 stabilizer 数
        self.num_t = mapper.mapping_info.num_t
        self.rounds = mapper.mapping_info.rounds
        
        # --- 注册 buffers ---
        # Z mapping
        self.register_buffer('gather_z', mapper.mapping_info.gather_z)
        self.register_buffer('valid_z', mapper.mapping_info.valid_z.view(self.num_t, self.num_z))
        self.register_buffer('z_neighbors', mapper.mapping_info.z_neighbors)
        self.register_buffer('z_hint_neighbors', mapper.mapping_info.z_hint_neighbors)
        # X mapping
        self.register_buffer('gather_x', mapper.mapping_info.gather_x)
        self.register_buffer('valid_x', mapper.mapping_info.valid_x.view(self.num_t, self.num_x))
        self.register_buffer('x_neighbors', mapper.mapping_info.x_neighbors)
        self.register_buffer('x_hint_neighbors', mapper.mapping_info.x_hint_neighbors)
        # 空间坐标 (用于 RoPE)
        self.register_buffer('spatial_coords', mapper.get_spatial_coords('both'))  # [num_z+num_x, 2]
        
        # --- Embedding (保持原有离散编码，X 和 Z 共享 embedding 表) ---
        self.emb_space = nn.Embedding(32, d_model)    # C*16 + TL*8 + TR*4 + BL*2 + BR
        self.emb_temp = nn.Embedding(4, d_model)      # T_prev*2 + T_curr
        self.emb_x_hints = nn.Embedding(16, d_model)  # H1*8 + H2*4 + H3*2 + H4 (异类型邻居hint)
        
        self.stem_norm = RMSNorm(d_model)
        self.stem_resnet = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model),
        )
        
        # --- RoPE ---
        self.rope_gen = CoordinateRoPE(d_model // n_heads)
        
        # --- 主干网络: AQ2 交替架构 ---
        # RNN → RNN → [3 TF] → RNN → [3 TF] → RNN → [3 TF] → RNN
        self.n_rnn = 5
        self.rnn_layers = nn.ModuleList([RecurrentBlock(d_model) for _ in range(self.n_rnn)])
        
        self.n_tf = 9
        self.tf_layers = nn.ModuleList([SpatialTransformerBlock(d_model, n_heads) for _ in range(self.n_tf)])
        
        # --- Readout (保持原有设计不变) ---
        self.logical_query_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.readout_layers = nn.ModuleList([AQCrossAttentionLayer(d_model, n_heads) for _ in range(2)])
        self.res_dense1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.res_dense2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def get_time_sinusoidal_encoding(self, num_t, d_model, device):
        position = torch.arange(num_t, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, num_t, 1, d_model, device=device)
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def _embed_stabilizers(self, det_raw, gather_idx, valid_mask, 
                           neighbors_same, hint_neighbors, num_spatial):
        """
        通用 embedding 函数，X 和 Z 共用相同逻辑。
        
        det_raw: [B, num_detectors] — 原始 detection event
        gather_idx: [num_t * num_spatial] — stim index
        valid_mask: [num_t, num_spatial]
        neighbors_same: [num_spatial, 4] — 同类型对角邻居 index (index into spatial)
        hint_neighbors: [num_t * num_spatial, 4] — 异类型邻居的 stim index
        num_spatial: int
        
        Returns: [B, num_t, num_spatial, d_model]
        """
        B = det_raw.shape[0]
        
        # 1. Gather 本 stabilizer 的 detection events → [B, num_t, num_spatial]
        X_3d = det_raw.gather(1, gather_idx.unsqueeze(0).expand(B, -1)).view(B, self.num_t, num_spatial)
        X_3d = (X_3d * valid_mask).long()
        
        # 2. 空间 embedding: C*16 + TL*8 + TR*4 + BL*2 + BR
        # 用 padding index 处理边界 (neighbors_same 的 padding 值 = num_spatial)
        X_sp = F.pad(X_3d, (0, 1), value=0)  # 在 spatial 维度末尾补 0
        N_vals = X_sp[:, :, neighbors_same]   # [B, num_t, num_spatial, 4]
        C = X_3d
        TL, TR, BL, BR = N_vals[..., 0], N_vals[..., 1], N_vals[..., 2], N_vals[..., 3]
        idx_space = C * 16 + TL * 8 + TR * 4 + BL * 2 + BR * 1
        
        # 3. 时间 embedding: T_prev*2 + T_curr
        X_t = F.pad(X_3d, (0, 0, 1, 0), value=0)  # 在时间维度前补 0
        T_prev = X_t[:, 0:self.num_t, :]
        T_curr = X_3d
        idx_temp = T_prev * 2 + T_curr * 1
        
        # 4. Hint embedding: H1*8 + H2*4 + H3*2 + H4 (异类型邻居)
        X_hints = det_raw.gather(1, hint_neighbors.view(-1).unsqueeze(0).expand(B, -1))
        X_hints = X_hints.view(B, self.num_t, num_spatial, 4).long()
        H1, H2, H3, H4 = X_hints[..., 0], X_hints[..., 1], X_hints[..., 2], X_hints[..., 3]
        idx_hint = H1 * 8 + H2 * 4 + H3 * 2 + H4 * 1
        
        # 5. 时间位置编码
        emb_time = self.get_time_sinusoidal_encoding(self.num_t, self.d_model, det_raw.device)
        
        # 6. 合成
        emb = (self.emb_space(idx_space) + self.emb_temp(idx_temp) + 
               self.emb_x_hints(idx_hint) + emb_time)
        
        emb = emb + self.stem_resnet(self.stem_norm(emb))
        return emb  # [B, num_t, num_spatial, d_model]

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        
        # ==================== Embedding ====================
        # Z stabilizers
        emb_z = self._embed_stabilizers(
            x,
            gather_idx=self.gather_z, valid_mask=self.valid_z,
            neighbors_same=self.z_neighbors, hint_neighbors=self.z_hint_neighbors,
            num_spatial=self.num_z
        )  # [B, num_t, num_z, D]
        
        # X stabilizers  
        emb_x = self._embed_stabilizers(
            x,
            gather_idx=self.gather_x, valid_mask=self.valid_x,
            neighbors_same=self.x_neighbors, hint_neighbors=self.x_hint_neighbors,
            num_spatial=self.num_x
        )  # [B, num_t, num_x, D]
        
        # 拼接 X 和 Z: [B, num_t, num_z + num_x, D]
        emb = torch.cat([emb_z, emb_x], dim=2)
        
        # Stabilizer dropout (训练时)
        if self.training and torch.rand(1).item() < 0.8:
            dropout_mask = (torch.rand(B, self.num_t, self.num_stab, 1, device=device) > 0.5).to(emb.dtype)
            emb = emb * dropout_mask * 2.0
        
        # RoPE 频率 (对整个 Z+X 序列)
        freqs_x, freqs_y = self.rope_gen.get_freqs(self.spatial_coords, device)
        
        # ==================== 主干: RNN + TF 交替 ====================
        # 初始化 RNN hidden states: [B, num_stab, D]
        rnn_states = [torch.zeros(B, self.num_stab, self.d_model, device=device) 
                      for _ in range(self.n_rnn)]
        
        all_time_feats = []
        
        for t in range(self.num_t):
            curr = emb[:, t]  # [B, num_stab, D]
            
            # --- Block 0: RNN_0 → RNN_1 ---
            curr, rnn_states[0] = self.rnn_layers[0](curr, rnn_states[0])
            curr, rnn_states[1] = self.rnn_layers[1](curr, rnn_states[1])
            
            # --- Block 1: TF_0, TF_1, TF_2 ---
            curr = self.tf_layers[0](curr, freqs_x, freqs_y)
            curr = self.tf_layers[1](curr, freqs_x, freqs_y)
            # curr = self.tf_layers[2](curr, freqs_x, freqs_y)
            
            # --- Block 2: RNN_2 ---
            curr, rnn_states[2] = self.rnn_layers[2](curr, rnn_states[2])
            
            # --- Block 3: TF_3, TF_4, TF_5 ---
            curr = self.tf_layers[2](curr, freqs_x, freqs_y)
            curr = self.tf_layers[3](curr, freqs_x, freqs_y)
            # curr = self.tf_layers[5](curr, freqs_x, freqs_y)
            
            # --- Block 4: RNN_3 ---
            curr, rnn_states[3] = self.rnn_layers[3](curr, rnn_states[3])
            
            # --- Block 5: TF_6, TF_7, TF_8 ---
            curr = self.tf_layers[4](curr, freqs_x, freqs_y)
            curr = self.tf_layers[5](curr, freqs_x, freqs_y)
            # curr = self.tf_layers[8](curr, freqs_x, freqs_y)
            
            # --- Block 6: RNN_4 ---
            curr, rnn_states[4] = self.rnn_layers[4](curr, rnn_states[4])
            
            all_time_feats.append(curr)
        
        # ==================== Readout (保持不变) ====================
        full_st_feat = torch.stack(all_time_feats, dim=1)  # [B, num_t, num_stab, D]
        full_st_feat_flat = full_st_feat.view(B, self.num_t * self.num_stab, self.d_model)
        
        # 联合 valid mask
        valid_z_2d = self.valid_z                    # [num_t, num_z]
        valid_x_2d = self.valid_x                    # [num_t, num_x]
        valid_all = torch.cat([valid_z_2d, valid_x_2d], dim=1)  # [num_t, num_stab]
        valid_mask_flat = valid_all.reshape(-1)       # [num_t * num_stab]
        
        padding_mask = (valid_mask_flat == 0).unsqueeze(0).expand(B, -1)
        mask_float = valid_mask_flat.view(1, -1, 1)
        den = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        pooled = (full_st_feat_flat * mask_float).sum(dim=1, keepdim=True) / den
        q = pooled + self.logical_query_embed.expand(B, -1, -1)
        
        for layer in self.readout_layers:
            q = layer(q, full_st_feat_flat, padding_mask)
            
        q = q + self.res_dense1(q)
        q = q + self.res_dense2(q)
        out = self.head(self.head_norm(q))
        
        return out.squeeze(-1).squeeze(-1)

    def forward_with_intermediates(self, x):
        """
        返回中间表征的前向传播，用于知识蒸馏。
        不执行 training dropout，收集每个时间步的 RNN/TF 中间状态。

        Args:
            x: [B, num_detectors] 原始平展检测事件

        Returns:
            logits: [B, 1]
            intermediates: dict
                'cnn_features': [B, num_t, num_stab, d_model] — TF 空间混合输出
                'decoder_states': [B, num_t, num_stab, d_model] — 最后一层 RNN 隐状态
                'readout_features': [B, d_model] — cross-attention 后的 query
                'readout_logits': [B, 1] — 最终 logits
        """
        B = x.shape[0]
        device = x.device

        # ==================== Embedding ====================
        emb_z = self._embed_stabilizers(
            x,
            gather_idx=self.gather_z, valid_mask=self.valid_z,
            neighbors_same=self.z_neighbors, hint_neighbors=self.z_hint_neighbors,
            num_spatial=self.num_z
        )
        emb_x = self._embed_stabilizers(
            x,
            gather_idx=self.gather_x, valid_mask=self.valid_x,
            neighbors_same=self.x_neighbors, hint_neighbors=self.x_hint_neighbors,
            num_spatial=self.num_x
        )
        emb = torch.cat([emb_z, emb_x], dim=2)

        # 推理时不使用 dropout

        freqs_x, freqs_y = self.rope_gen.get_freqs(self.spatial_coords, device)

        # ==================== 主干: RNN + TF 交替 ====================
        rnn_states = [torch.zeros(B, self.num_stab, self.d_model, device=device)
                      for _ in range(self.n_rnn)]

        all_time_feats = []
        all_decoder_states = []

        for t in range(self.num_t):
            curr = emb[:, t]

            # 记录进入本时间步前最后一层 RNN 的隐状态
            all_decoder_states.append(rnn_states[4].clone())

            # --- Block 0: RNN_0 → RNN_1 ---
            curr, rnn_states[0] = self.rnn_layers[0](curr, rnn_states[0])
            curr, rnn_states[1] = self.rnn_layers[1](curr, rnn_states[1])

            # --- Block 1: TF_0, TF_1 ---
            curr = self.tf_layers[0](curr, freqs_x, freqs_y)
            curr = self.tf_layers[1](curr, freqs_x, freqs_y)

            # --- Block 2: RNN_2 ---
            curr, rnn_states[2] = self.rnn_layers[2](curr, rnn_states[2])

            # --- Block 3: TF_2, TF_3 ---
            curr = self.tf_layers[2](curr, freqs_x, freqs_y)
            curr = self.tf_layers[3](curr, freqs_x, freqs_y)

            # --- Block 4: RNN_3 ---
            curr, rnn_states[3] = self.rnn_layers[3](curr, rnn_states[3])

            # --- Block 5: TF_4, TF_5 ---
            curr = self.tf_layers[4](curr, freqs_x, freqs_y)
            curr = self.tf_layers[5](curr, freqs_x, freqs_y)

            # --- Block 6: RNN_4 ---
            curr, rnn_states[4] = self.rnn_layers[4](curr, rnn_states[4])

            all_time_feats.append(curr)

        # [B, num_t, num_stab, d_model]
        cnn_features = torch.stack(all_time_feats, dim=1)
        decoder_states = torch.stack(all_decoder_states, dim=1)

        # ==================== Readout ====================
        full_st_feat_flat = cnn_features.view(B, self.num_t * self.num_stab, self.d_model)

        valid_z_2d = self.valid_z
        valid_x_2d = self.valid_x
        valid_all = torch.cat([valid_z_2d, valid_x_2d], dim=1)
        valid_mask_flat = valid_all.reshape(-1)

        padding_mask = (valid_mask_flat == 0).unsqueeze(0).expand(B, -1)
        mask_float = valid_mask_flat.view(1, -1, 1)
        den = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)

        pooled = (full_st_feat_flat * mask_float).sum(dim=1, keepdim=True) / den
        q = pooled + self.logical_query_embed.expand(B, -1, -1)

        for layer in self.readout_layers:
            q = layer(q, full_st_feat_flat, padding_mask)

        q = q + self.res_dense1(q)
        q = q + self.res_dense2(q)

        readout_features = q.squeeze(1)  # [B, d_model]

        out = self.head(self.head_norm(q))
        logits = out.view(B, 1)

        intermediates = {
            'cnn_features': cnn_features,
            'decoder_states': decoder_states,
            'readout_features': readout_features,
            'readout_logits': logits,
        }

        return logits, intermediates


# ==============================================================================
# 数据集 (使用新的 FullMapper)
# ==============================================================================
class OnlineSurfaceCodeDataset(IterableDataset):
    def __init__(self, d, p, batch_size, rank=0, world_size=1):
        self.d = d; self.p = p; self.batch_size = batch_size
        self.rank = rank; self.world_size = world_size
        self.mapper = FullMapper(d, d)
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p
        )
        self.sampler = self.circuit.compile_detector_sampler()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        entropy = int.from_bytes(os.urandom(4), byteorder="little")
        time_ns = time.time_ns()
        final_seed = (time_ns + entropy + worker_id * 10000 + self.rank * 100000) % (2**32 - 1)
        np.random.seed(final_seed)
        torch.manual_seed(final_seed)
        sampler = self.circuit.compile_detector_sampler(seed=final_seed)
        while True:
            det, obs = sampler.sample(self.batch_size, separate_observables=True)
            x = torch.from_numpy(det).float()
            y = torch.from_numpy(obs).float()
            yield x, y
            
    def get_info(self): return None, None, self.mapper


# ==============================================================================
# 训练工具函数 (保持不变)
# ==============================================================================
def set_freeze_mode(model, mode="freeze_logic", rank=0):
    if rank == 0: print(f"--> [Model Status] Switching to: {mode}")
    for name, param in model.named_parameters():
        if mode == "unfreeze_all": param.requires_grad = True
        elif mode == "freeze_logic":
            if "input_stem" in name: param.requires_grad = False
            elif "mlp" in name: param.requires_grad = False 
            elif "attn" in name or "qkv" in name or "proj" in name: param.requires_grad = True
            elif "norm" in name or "head" in name: param.requires_grad = True
            else: param.requires_grad = False 
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--> [Info] Trainable Params: {trainable/1e6:.2f}M")

def log_layer_gradients(model, step, rank):
    if rank != 0: return
    norms = {"Stem":0, "RNN":0, "TF":0, "Head":0}
    counts = {"Stem":0, "RNN":0, "TF":0, "Head":0}
    for n, p in model.named_parameters():
        if p.grad is not None:
            val = p.grad.data.norm(2).item()
            k = "Stem" if "stem" in n or "emb" in n else "Head" if "head" in n or "readout" in n else "RNN" if "rnn" in n or "gru" in n else "TF"
            norms[k] += val; counts[k] += 1
    s = f"[Grad {step}] "
    for k in norms: 
        if counts[k] > 0: s += f"{k}: {norms[k]/counts[k]:.4f} | "
    print(s)


# ==============================================================================
# 训练主循环
# ==============================================================================
def run_training(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0: 
        print(f"\n=== ALPHA-QUBIT V2: d={args.d} | X+Z Joint | RNN+TF Interleaved ===")

    temp_ds = OnlineSurfaceCodeDataset(args.d, args.train_p, 1, rank=rank, world_size=world_size)
    mapper = temp_ds.get_info()[2]
    
    if rank == 0:
        print(f"    Z stabilizers: {mapper.mapping_info.num_z}")
        print(f"    X stabilizers: {mapper.mapping_info.num_x}")
        print(f"    Total per step: {mapper.mapping_info.num_z + mapper.mapping_info.num_x}")
        print(f"    Time steps: {mapper.mapping_info.num_t}")
    
    train_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.train_p, args.batch_size, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=8, pin_memory=True
    )
    val_batch = min(args.batch_size, 128)
    val_high_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.train_p, val_batch, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=4
    )
    val_low_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.eval_p, val_batch, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=4
    )

    model = AlphaQubitV2(mapper, d_model=256, n_heads=8).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Total parameters: {total_params / 1e6:.2f}M")
    
    start_step = 0
    saved_d = -1
    if args.resume and os.path.exists(args.resume):
        if rank == 0: print(f"--> Loading Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False) 
        raw_state_dict = ckpt['model_state']
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            new_k = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[new_k] = v
        saved_d = ckpt.get('d', -1)
        if saved_d != args.d:
            if rank == 0: print(f"--> Transfer detected (d={saved_d} -> {args.d}). Dropping Coord/Mapper buffers.")
            exclude_keys = ['coords', 'freq', 'mapper', 'mask', 'neighbors', 'gather', 'valid', 'hint']
            clean_state_dict = {
                k: v for k, v in clean_state_dict.items() 
                if not any(ex in k for ex in exclude_keys)
            }
            start_step = 0
        else:
            start_step = ckpt.get('step', 0)
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        if rank == 0: print(f"--> Training from scratch (no checkpoint)")
            
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    set_freeze_mode(model, "unfreeze_all", rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, fused=True)
    
    def lr_lambda(current_step):
        warmup_steps = 3000; decay_steps = args.max_steps; min_lr_ratio = 0.05 
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        if current_step < decay_steps:
            progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.train(); ACCUM = max(1, 1024 // (args.batch_size * world_size)); micro = 0
    iterator = iter(train_loader)
    update_step = start_step
    metrics = torch.zeros(2, device=device)
    
    while update_step < args.max_steps:
        try: X, Y = next(iterator)
        except: iterator = iter(train_loader); X, Y = next(iterator)
        
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        
        my_context = model.no_sync() if dist.is_initialized() and (micro + 1) % ACCUM != 0 else contextlib.nullcontext()
        
        with my_context:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(X) 
                loss = nn.BCEWithLogitsLoss()(pred.view(-1, 1), Y) / ACCUM
            
            loss.backward()

        micro += 1
        
        if micro % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if update_step % 5000 == 0: log_layer_gradients(model, update_step, rank)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()
            
            if update_step % 1000 == 0:
                acc_high = validate(model, val_high_loader, device)
                acc_low = validate(model, val_low_loader, device)
                metrics = torch.tensor([acc_high, acc_low], device=device)
                if dist.is_initialized(): dist.all_reduce(metrics, op=dist.ReduceOp.SUM); metrics /= world_size
                if rank == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {update_step} | LR {lr:.2e} | High: {metrics[0]:.5f} | Low: {metrics[1]:.5f}", flush=True)
                    torch.save({'model_state': (model.module if dist.is_initialized() else model).state_dict(), 
                                'd': args.d, 'step': update_step}, args.output)
                if metrics[0] >= args.target_high and metrics[1] >= args.target_low:
                        if rank == 0:
                            print(f"\n🎉🎉🎉 [SUCCESS] Target Reached at Step {update_step}! High: {metrics[0]:.5f}, Low: {metrics[1]:.5f} 🎉🎉🎉")
                            print("--> Stopping training early to save compute.")
                        break
            update_step += 1
    cleanup_ddp()

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i > 50: break
            X, Y = X.to(device), Y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = (model(X) > 0).float().view(-1, 1)
            correct += (pred == Y).float().sum().item()
            total += X.size(0)
    model.train()
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--train_p", type=float, default=0.008)
    parser.add_argument("--eval_p", type=float, default=0.005)
    parser.add_argument("--target_high", type=float, required=True)
    parser.add_argument("--target_low", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    try:
        run_training(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
