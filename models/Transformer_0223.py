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

# --- 物理映射层 (Strict Z-Only) ---
@dataclass
class ZMappingInfo:
    valid_mask: torch.Tensor
    gather_indices: torch.Tensor
    gather_x_neighbors: torch.Tensor
    num_z_spatial: int
    num_t: int
    rounds: int

class StrictZMapper(nn.Module):
    def __init__(self, d: int, rounds: int):
        super().__init__()
        self.d = d
        self.rounds = rounds
        self.circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds)
        self.mapping_info = self._build_strict_z_mapping()
        
        self.register_buffer('gather_indices', self.mapping_info.gather_indices)
        self.register_buffer('valid_mask', self.mapping_info.valid_mask)
        self.register_buffer('gather_x_neighbors', self.mapping_info.gather_x_neighbors)

    def _build_strict_z_mapping(self) -> ZMappingInfo:
        coords = self.circuit.get_detector_coordinates()
        if not coords: 
            return ZMappingInfo(torch.empty(0), torch.empty(0), torch.empty(0), 0, 0, self.rounds)
        
        max_t = max([t for _, _, t in coords.values()])
        
        # 1. 精准分离 Z 和 X 的空间坐标 (使用 r=y/2, c=x/2 网格)
        z_locs = sorted(list(set((int(round(y/2)), int(round(x/2))) 
                                 for idx, (x, y, t) in coords.items() if math.isclose(t, max_t))),
                        key=lambda p: (p[0], p[1]))
                        
        loc_to_idx_z = {loc: i for i, loc in enumerate(z_locs)}
        
        # 找出所有不属于 Z 位置的坐标，它们就是 X 探测器！
        x_locs_set = set()
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y/2)), int(round(x/2))
            if (r, c) not in loc_to_idx_z:
                x_locs_set.add((r, c))
                
        # 2. 收集时间序列
        found_dets = []
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y/2)), int(round(x/2))
            if (r, c) in loc_to_idx_z:
                found_dets.append((t, r, c, idx))
                
        if not found_dets:
             return ZMappingInfo(torch.empty(0), torch.empty(0), torch.empty(0), 0, 0, self.rounds)

        unique_times = sorted(list(set(t for t, _, _, _ in found_dets)))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        num_t = len(unique_times)
        num_spatial = len(z_locs)
        
        # 建立时空反查字典 (r, c, t) -> Stim Index
        coord_time_to_idx = {}
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y/2)), int(round(x/2))
            coord_time_to_idx[(r, c, t)] = idx
            
        # 3. 初始化矩阵
        gather_matrix = torch.zeros((num_t, num_spatial), dtype=torch.long)
        valid_mask = torch.zeros((num_t, num_spatial), dtype=torch.float32) 
        gather_x_neighbors = torch.zeros((num_t, num_spatial, 4), dtype=torch.long) # [新增]
        
        # 4. 为每个 Z 计算 4 个最近的 X 邻居
        z_to_x_locs = {}
        for zr, zc in z_locs:
            sorted_x = sorted(list(x_locs_set), key=lambda p: (p[0]-zr)**2 + (p[1]-zc)**2)
            z_to_x_locs[(zr, zc)] = sorted_x[:4]
            
        # 5. 填装数据
        for t_idx, t in enumerate(unique_times):
            for z_idx, (zr, zc) in enumerate(z_locs):
                if (zr, zc, t) in coord_time_to_idx:
                    gather_matrix[t_idx, z_idx] = coord_time_to_idx[(zr, zc, t)]
                    valid_mask[t_idx, z_idx] = 1.0
                
                for i, (xr, xc) in enumerate(z_to_x_locs[(zr, zc)]):
                    if (xr, xc, t) in coord_time_to_idx:
                        gather_x_neighbors[t_idx, z_idx, i] = coord_time_to_idx[(xr, xc, t)]
                        
        return ZMappingInfo(
            valid_mask=valid_mask.flatten(), 
            gather_indices=gather_matrix.flatten(), 
            gather_x_neighbors=gather_x_neighbors.view(-1, 4), # reshape 使其一维对齐
            num_z_spatial=num_spatial, 
            num_t=num_t, 
            rounds=self.rounds
        )

    def get_spatial_coords(self):
        coords = self.circuit.get_detector_coordinates()
        if self.mapping_info.num_z_spatial == 0: return torch.zeros(0, 2)
        
        # 只取第一轮的 gather indices 来解析空间坐标
        first_round_indices = self.mapping_info.gather_indices[:self.mapping_info.num_z_spatial].tolist()
        spatial_xy = []
        BASE_D = 7.0
        for idx in first_round_indices:
            # Handle potential padding 0s if using valid_mask
            if idx in coords:
                x, y, _ = coords[idx]
            else:
                x, y = 0, 0 # Fallback
            nx = x / (2 * BASE_D) * 2 - 1
            ny = y / (2 * BASE_D) * 2 - 1
            spatial_xy.append([nx, ny])
        return torch.tensor(spatial_xy, dtype=torch.float32)

    def get_full_coords(self):
        coords = self.circuit.get_detector_coordinates()
        all_indices = self.mapping_info.gather_indices.tolist()
        full_xyt = []
        BASE_D = 7.0
        BASE_ROUNDS = 7.0
        for idx in all_indices:
            if idx in coords:
                x, y, t = coords[idx]
            else:
                x, y, t = 0, 0, 0
            nx = x / (2 * BASE_D) * 2 - 1
            ny = y / (2 * BASE_D) * 2 - 1
            nt = t / (BASE_ROUNDS)
            full_xyt.append([nx, ny, nt])
        return torch.tensor(full_xyt, dtype=torch.float32)

    def get_z_neighbors(self):
        """
        获取每个 Z-detector 的 4 个对角线邻居 (TL, TR, BL, BR) 的索引。
        返回 shape: [num_z, 4]
        """
        coords = self.circuit.get_detector_coordinates()
        if not coords: return torch.empty(0, 4, dtype=torch.long)
        
        max_t = max([t for _, _, t in coords.values()])
        z_locs = sorted(list(set((int(round(y/2)), int(round(x/2))) 
                                 for idx, (x, y, t) in coords.items() if math.isclose(t, max_t))),
                        key=lambda p: (p[0], p[1]))
        
        loc_to_idx = {loc: i for i, loc in enumerate(z_locs)}
        num_z = len(z_locs)
        
        neighbors = torch.full((num_z, 4), num_z, dtype=torch.long) 
        
        for i, (r, c) in enumerate(z_locs):
            for j, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                if (r + dr, c + dc) in loc_to_idx:
                    neighbors[i, j] = loc_to_idx[(r + dr, c + dc)]
                    
        return neighbors

# --- 组件 ---
class CoordinateRoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
        self.head_dim = head_dim
        self.quarter_dim = head_dim // 4 

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.quarter_dim, dtype=torch.float32) / self.quarter_dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_freqs(self, coords, device):
        x = coords[:, 0].to(device)
        y = coords[:, 1].to(device)

        fx = torch.einsum("i,j->ij", x, self.inv_freq)
        fy = torch.einsum("i,j->ij", y, self.inv_freq)

        half_freqs = torch.cat([fx, fy], dim=-1)
        
        return torch.cat([half_freqs, half_freqs], dim=-1)

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

class AQTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim); self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = SwiGLU(dim, 4 * dim)

    def forward(self, x, prev_state=None, freqs=None):
        x_in = x + prev_state if prev_state is not None else x
        qkv = self.qkv(self.norm1(x_in))
        q, k, v = map(lambda t: t.view(x.size(0), x.size(1), self.n_heads, self.head_dim), qkv.chunk(3, dim=-1))
        
        # RoPE Application
        if freqs is not None:
            freqs = freqs.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, HeadDim]
            # Rotate Half Strategy
            q_rot = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
            k_rot = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
            q = q * freqs.cos() + q_rot * freqs.sin()
            k = k * freqs.cos() + k_rot * freqs.sin()
            
        out = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
        x = x_in + self.proj(out.transpose(1, 2).reshape(x.size(0), x.size(1), self.dim))
        return x + self.mlp(self.norm2(x)), x

class AQCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, q, kv, padding_mask):
        # Pre-Norm
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        
        # Cross Attention
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, key_padding_mask=padding_mask)
        q = q + attn_out
        
        # MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q

# --- 主模型 ---
class AlphaQubitStrictZ(nn.Module):
    def __init__(self, mapper, d_model=256, n_layers=6, n_heads=8):
        super().__init__()
        self.d_model = d_model; self.n_layers = n_layers
        
        self.num_z = mapper.mapping_info.num_z_spatial
        self.num_t = mapper.mapping_info.num_t
        self.rounds = mapper.rounds
        
        self.register_buffer('spatial_coords', mapper.get_spatial_coords()) 
        self.register_buffer('valid_mask', mapper.mapping_info.valid_mask.view(self.num_t, self.num_z))
        self.register_buffer('z_neighbors', mapper.get_z_neighbors())
        self.register_buffer('gather_indices', mapper.mapping_info.gather_indices)
        self.register_buffer('gather_x_neighbors', mapper.mapping_info.gather_x_neighbors)

        self.emb_space = nn.Embedding(32, d_model)
        self.emb_temp = nn.Embedding(4, d_model)
        self.emb_x_hints = nn.Embedding(16, d_model)
        
        self.stem_norm = RMSNorm(d_model)
        self.stem_resnet = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.rope_gen = CoordinateRoPE(d_model // n_heads)
        self.layers = nn.ModuleList([AQTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.logical_query_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.readout_layers = nn.ModuleList([AQCrossAttentionLayer(d_model, n_heads) for _ in range(2)])
        self.res_dense1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.res_dense2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def get_time_sinusoidal_encoding(self, num_t, d_model, device):
        position = torch.arange(num_t, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, num_t, 1, d_model, device=device) # 形状匹配 [1, num_t, 1, d_model]
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B = x.shape[0]
        X_3d_z = x.gather(1, self.gather_indices.unsqueeze(0).expand(B, -1)).view(B, self.num_t, self.num_z)
        X_3d_z = (X_3d_z * self.valid_mask).long()
        X_x_neighbors = x.gather(1, self.gather_x_neighbors.view(-1).unsqueeze(0).expand(B, -1)).view(B, self.num_t, self.num_z, 4).long()
        
        X_sp = F.pad(X_3d_z, (0, 1), value=0)
        N_vals = X_sp[:, :, self.z_neighbors]
        C = X_3d_z
        TL, TR, BL, BR = N_vals[..., 0], N_vals[..., 1], N_vals[..., 2], N_vals[..., 3]
        idx_space = C * 16 + TL * 8 + TR * 4 + BL * 2 + BR * 1
        
        X_t = F.pad(X_3d_z, (0, 0, 1, 0), value=0) 
        T_prev = X_t[:, 0:self.num_t, :]
        T_curr = X_3d_z
        idx_temp = T_prev * 2 + T_curr * 1
        
        X1, X2, X3, X4 = X_x_neighbors[..., 0], X_x_neighbors[..., 1], X_x_neighbors[..., 2], X_x_neighbors[..., 3]
        idx_x_hints = X1 * 8 + X2 * 4 + X3 * 2 + X4 * 1

        emb_time = self.get_time_sinusoidal_encoding(self.num_t, self.d_model, x.device)

        emb_s = self.emb_space(idx_space)
        emb_t = self.emb_temp(idx_temp)  
        emb_x = self.emb_x_hints(idx_x_hints)
        
        # 完美融合：空间拓扑 + 因果时序导数 + X关联提示 + 绝对时间戳
        emb = emb_s + emb_t + emb_x + emb_time
        emb = emb + self.stem_resnet(self.stem_norm(emb))

        if self.training and torch.rand(1).item() < 0.8:
            dropout_mask = (torch.rand(B, self.num_t, self.num_z, 1, device=x.device) > 0.5).to(emb.dtype)
            emb = emb * dropout_mask * 2.0
        
        freqs = self.rope_gen.get_freqs(self.spatial_coords, x.device) 
        
        states = [torch.zeros(B, self.num_z, self.d_model, device=x.device) for _ in range(self.n_layers)]
        
        # 【修复3：收集所有时间步的特征，彻底解决灾难性遗忘】
        all_time_feats = []
        
        for t in range(self.num_t):
            curr = emb[:, t]
            new_states = []
            for i, layer in enumerate(self.layers):
                curr, st = layer(curr, states[i], freqs)
                new_states.append(st)
            states = new_states
            all_time_feats.append(curr) # 保存每一轮的输出特征
            
        # 堆叠成完整的时空张量: [B, num_t, num_z, d_model]
        full_st_feat = torch.stack(all_time_feats, dim=1)
        # 展平为完整的 Sequence 以供 Readout 使用: [B, num_t * num_z, d_model]
        full_st_feat_flat = full_st_feat.view(B, self.num_t * self.num_z, self.d_model)
            
        # ==================== Readout 改造 ====================
        valid_mask_flat = self.valid_mask.view(-1).to(x.device) # [num_t * num_z]
        padding_mask = (valid_mask_flat == 0).unsqueeze(0).expand(B, -1) # [B, num_t * num_z]
        
        mask_float = valid_mask_flat.view(1, -1, 1) # [1, num_t * num_z, 1]
        den = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        # 对整个时空的所有有效节点进行 Mean Pooling
        pooled = (full_st_feat_flat * mask_float).sum(dim=1, keepdim=True) / den # [B, 1, D]
        
        q = pooled + self.logical_query_embed.expand(B, -1, -1) 
        
        for layer in self.readout_layers:
            # Cross Attention 现在能纵观整个实验周期的所有特征！
            q = layer(q, full_st_feat_flat, padding_mask)
            
        q = q + self.res_dense1(q)
        q = q + self.res_dense2(q)
        out = self.head(self.head_norm(q))
        
        return out.squeeze(-1).squeeze(-1)

class OnlineSurfaceCodeDataset(IterableDataset):
    def __init__(self, d, p, batch_size, rank=0, world_size=1):
        self.d = d; self.p = p; self.batch_size = batch_size
        self.rank = rank; self.world_size = world_size
        self.mapper = StrictZMapper(d, d) 
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
        self.z_indices = self.mapper.mapping_info.gather_indices.numpy()
        self.z_mask = self.mapper.mapping_info.valid_mask.numpy()

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
            # z_det = det[:, self.z_indices]
            # z_det = z_det * self.z_mask[None, :]
            # x = self.soft_sim.transform(torch.from_numpy(z_det).float())
            x = torch.from_numpy(det).float()
            y = torch.from_numpy(obs).float()
            yield x, y
            
    def get_info(self): return None, None, self.mapper

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
    norms = {"Stem":0, "Attn":0, "MLP":0, "Head":0}; counts = {"Stem":0, "Attn":0, "MLP":0, "Head":0}
    for n, p in model.named_parameters():
        if p.grad is not None:
            val = p.grad.data.norm(2).item()
            k = "Stem" if "stem" in n else "Head" if "head" in n else "Attn" if "attn" in n or "qkv" in n else "MLP"
            norms[k]+=val; counts[k]+=1
    s = f"[Grad {step}] "
    for k in norms: 
        if counts[k]>0: s += f"{k}: {norms[k]/counts[k]:.4f} | "
    print(s)

def run_training(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0: 
        print(f"\n=== ALPHA-QUBIT STRICT Z-ONLY: d={args.d} | Recurrent | Fourier ===")

    temp_ds = OnlineSurfaceCodeDataset(args.d, args.train_p, 1, rank=rank, world_size=world_size)
    mapper = temp_ds.get_info()[2]
    
    train_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.train_p, 256, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=8, pin_memory=True
    )
    val_high_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.train_p, 256, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=4
    )
    val_low_loader = DataLoader(
        OnlineSurfaceCodeDataset(args.d, args.eval_p, 256, rank=rank, world_size=world_size), 
        batch_size=None, num_workers=4
    )

    model = AlphaQubitStrictZ(mapper, d_model=256, n_layers=6, n_heads=8).to(device)
    # model = SimpleMLPBaseline(mapper).to(device)
    
    start_step = 0
    saved_d = -1
    if args.resume and os.path.exists(args.resume):
        if rank == 0: print(f"--> Loading Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device) 
        raw_state_dict = ckpt['model_state']
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            new_k = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[new_k] = v
        saved_d = ckpt.get('d', -1)
        if saved_d != args.d:
            if rank == 0: print(f"--> Transfer detected (d={saved_d} -> {args.d}). Dropping Coord buffers.")
            exclude_keys = ['coords', 'freq', 'mapper', 'mask', 'neighbors', 'gather']
            clean_state_dict = {
                k: v for k, v in clean_state_dict.items() 
                if not any(ex in k for ex in exclude_keys)
            }
            start_step = 0
        else:
            start_step = ckpt.get('step', 0)
        model.load_state_dict(clean_state_dict, strict=False)
            
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    is_transfer = (args.resume != "") and (saved_d != args.d) and (saved_d != -1)
    # if is_transfer: set_freeze_mode(model, "freeze_logic", rank)
    # else: set_freeze_mode(model, "unfreeze_all", rank)
    set_freeze_mode(model, "unfreeze_all", rank)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2, fused=True)
    
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
    UNFREEZE_STEP = int(args.max_steps * 0.2); has_unfrozen = not is_transfer
    
    for step in range(start_step, args.max_steps):
        try: X, Y = next(iterator)
        except: iterator = iter(train_loader); X, Y = next(iterator)
        
        # if is_transfer and not has_unfrozen and step >= UNFREEZE_STEP:
        #     if rank == 0: print(f"\n--> [Trigger] Step {step}: Unfreezing!")
        #     set_freeze_mode(model, "unfreeze_all", rank)
        #     has_unfrozen = True
        
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(X) 
            loss = nn.BCEWithLogitsLoss()(pred.view(-1, 1), Y) / ACCUM
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        micro += 1
        
        if micro % ACCUM == 0:
            if step % 1000 == 0: log_layer_gradients(model, step, rank)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()
            
            if step % 1000 == 0:
                acc_high = validate(model, val_high_loader, device)
                acc_low = validate(model, val_low_loader, device)
                metrics = torch.tensor([acc_high, acc_low], device=device)
                if dist.is_initialized(): dist.all_reduce(metrics, op=dist.ReduceOp.SUM); metrics /= world_size
                if rank == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {step} | LR {lr:.2e} | High: {metrics[0]:.5f} | Low: {metrics[1]:.5f}", flush=True)
                    torch.save({'model_state': (model.module if dist.is_initialized() else model).state_dict(), 
                                'd': args.d, 'step': step}, args.output)
            if metrics[0] >= args.target_high and metrics[1] >= args.target_low:
                    if rank == 0:
                        print(f"\n🎉🎉🎉 [SUCCESS] Target Reached at Step {step}! High: {metrics[0]:.5f}, Low: {metrics[1]:.5f} 🎉🎉🎉")
                        print("--> Stopping training early to save compute.")
                    
                    break
    cleanup_ddp()

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i > 20: break
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
    run_training(args)