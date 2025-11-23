from args import args
import torch
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import CRADLE, Data, mod2 # your existing module provides these
import os
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
# ------------------------------------------------------------
# Round Positional Encoding (RPE) — toggleable add-on
# ------------------------------------------------------------
class RoundPositionalProjector(nn.Module):
    """
    Explicit 'round-positional' encoding for flattened syndromes arranged as:
      [round1 | round2 | ... | round_r], each round contains D detector bits.
    For each position (detector_id, round_id), it produces a learnable scalar bias
    that is additively injected into the raw bits. This keeps input length unchanged
    and remains fully compatible with CRADLE’s interface.
    """
    def __init__(self, num_detectors: int, max_rounds: int, dim: int = 16, alpha_init: float = 1e-2):
        super().__init__()
        self.num_detectors = num_detectors
        self.max_rounds = max_rounds
        self.dim = dim
        self.det_emb = nn.Embedding(num_detectors, dim)
        self.rnd_emb = nn.Embedding(max_rounds + 1, dim) # round indices start at 1; index 0 is a fallback
        self.proj = nn.Linear(dim, 1, bias=False) # project embedding to scalar bias
        self.alpha = nn.Parameter(torch.tensor(alpha_init)) # global learnable scale

        nn.init.normal_(self.det_emb.weight, std=0.02)
        nn.init.normal_(self.rnd_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)

    @torch.no_grad()
    def _build_index_maps(self, syn_len: int, r: int):
        """
        Given flattened length syn_len = D * r, produce detector_id and round_id
        indices for each position.
        """
        assert r > 0 and syn_len % r == 0, f"syn_len={syn_len} must be divisible by r={r}"
        D = syn_len // r
        det_ids = torch.arange(syn_len) % D
        round_ids = torch.arange(syn_len) // D + 1 # 1..r
        return det_ids, round_ids, D

    def forward(self, syn_bits: torch.Tensor, r_list: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            syn_bits: (B, syn_len). Only the syndrome segment (floats in [0,1] or {0,1}).
            r_list:   (B,) number of rounds per sample.
            mask:     (B, syn_len) with 1 for valid and 0 for padded positions; if None, all valid.

        Returns:
            syn_out:  (B, syn_len) with additive round-positional bias injected on valid positions.
        """
        B, syn_len = syn_bits.shape
        device = syn_bits.device
        dtype = syn_bits.dtype

        if mask is None:
            mask = torch.ones_like(syn_bits, dtype=dtype, device=device)

        uniq = torch.unique(r_list)
        if uniq.numel() == 1:
            # Fast path: all samples share the same r
            r = int(uniq.item())
            det_ids, round_ids, _ = self._build_index_maps(syn_len, r)
            det_e = self.det_emb(det_ids.to(device))
            rnd_e = self.rnd_emb(round_ids.clamp_max(self.max_rounds).to(device))
            pe = self.proj(det_e + rnd_e).squeeze(-1) # (syn_len,)
            out = syn_bits + self.alpha * pe
            out = out * mask + syn_bits * (1 - mask)
            return out
        else:
            # General path: per-sample r (slightly slower)
            outs = []
            for b in range(B):
                r = int(r_list[b].item())
                det_ids, round_ids, _ = self._build_index_maps(syn_len, r)
                det_e = self.det_emb(det_ids.to(device))
                rnd_e = self.rnd_emb(round_ids.clamp_max(self.max_rounds).to(device))
                pe = self.proj(det_e + rnd_e).squeeze(-1)
                out_b = syn_bits[b] + self.alpha * pe
                if mask is not None:
                    out_b = out_b * mask[b] + syn_bits[b] * (1 - mask[b])
                outs.append(out_b.unsqueeze(0))
            return torch.cat(outs, dim=0)

# ---- Round Positional Encoding (RPE) hyperparameters (read from args.py) ----
USE_ROUND_PE        = getattr(args, 'use_round_pe', False)
ROUND_PE_DIM        = getattr(args, 'round_pe_dim', 16)
ROUND_PE_MAX_ROUNDS = getattr(args, 'round_pe_max_rounds', 16)
ROUND_PE_ALPHA_INIT = getattr(args, 'round_pe_alpha_init', 1e-2)

# ------------------------------------------------------------
# Existing configuration (kept as in your code)
# ------------------------------------------------------------

# We have "odd shot" and "even shot".
# DEM path example uses "even_for_odd"; evaluation for that should be on odd shots.

# Data: Google Surface Code data (X/Z error). Here we use surface code with distance d and rounds r.
basis, d = 'Z', 3
center = '5_3'
# r can be 3, 5, ..., up to e.g., 25 in Google data
r_values = [3, 5, 7, 9, 11, 13]

# How to handle varying r during training
# TRAINING_STRATEGY ∈ {'random', 'round_robin', 'curriculum'}
TRAINING_STRATEGY = 'curriculum'
CURRICULUM_WARMUP_EPOCHS = 10000 # number of warmup epochs for curriculum learning

# Whether to append r as an explicit input feature
USE_R_FEATURE = False
# If True: input = [syndrome, r_feature, logical], where r_feature = r / 10.0
# If False: input = [padded_syndrome, logical]. The model must infer r implicitly.

# [新增] 1. 从 args 获取 shot 类型
SHOT_TYPE = getattr(args, 'shot_type', 'odd_for_even')

def forward(n_s, m, van, syndrome, device, dtype, k=1):
    """
    Wrapper for your model's partial_forward inference step.
    Maps condition to [-1, 1], runs partial forward, maps back to [0,1],
    and returns the slice corresponding to the logical prediction.
    """
    condition = syndrome * 2 - 1
    x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1) / 2
    x = x[:, m:m + int(2 * k)]
    return x

def get_r_for_epoch(epoch, strategy, r_values, warmup_epochs):
    """
    Select r based on the chosen training strategy.

    Strategies:
      - 'random': random r for each epoch (could also do per-batch randomization).
      - 'round_robin': deterministic cycle over r_values.
      - 'curriculum': start from small r and progress to larger r; after warmup, switch to random.
    """
    if strategy == 'random':
        r_idx = torch.randint(0, len(r_values), (1,)).item()
        return r_values[r_idx]
    elif strategy == 'round_robin':
        r_idx = epoch % len(r_values)
        return r_values[r_idx]
    elif strategy == 'curriculum':
        if epoch < warmup_epochs:
            epochs_per_r = max(1, warmup_epochs // len(r_values))
            r_idx = (epoch // epochs_per_r) % len(r_values)
            return r_values[r_idx]
        else:
            r_idx = torch.randint(0, len(r_values), (1,)).item()
            return r_values[r_idx]
    else:
        raise ValueError(f"Unknown training strategy: {strategy}. Choose from: 'random', 'round_robin', 'curriculum'")

# [修改] 2. 更新 print_strategy_info 函数定义，使其接受 shot_type
def print_strategy_info(strategy, warmup_epochs, total_epochs, r_values, use_r_feature, shot_type):
    """
    Pretty-print the training configuration for clarity in logs.
    """
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Strategy : {strategy.upper()}")
    print(f"R Feature: {'ENABLED' if use_r_feature else 'DISABLED'}")
    print(f"Round-PE : {'ENABLED' if USE_ROUND_PE else 'DISABLED'}")
    # [新增] 打印 SHOT_TYPE
    print(f"Shot Type: {shot_type.upper()} (DEM path & model name)")
    print("-" * 70)
    if strategy == 'random':
        print("Each epoch randomly selects r from the provided r_values.")
    elif strategy == 'round_robin':
        print("Deterministic cycle through r_values each epoch.")
    elif strategy == 'curriculum':
        epochs_per_r = max(1, warmup_epochs // len(r_values))
        print("Curriculum: start easy (small r) → harder (large r) → random.")
        print(f"Warmup phase: {warmup_epochs:,} epochs (~{epochs_per_r:,} per r)")
        print(f"Random phase: {total_epochs - warmup_epochs:,} epochs")
    print("-" * 70)
    if use_r_feature:
        print("Input layout : [syndrome, r_feature, logical]  (r_feature = r/10.0)")
        print("Model sees r explicitly and can use it for generalization.")
    else:
        print("Input layout : [padded_syndrome, logical]")
        print("Model must infer r from the padded pattern; more challenging but compact.")
    print("=" * 70)

def maybe_apply_round_pe(syndromes: torch.Tensor, r: int,
                         projector: Optional[RoundPositionalProjector],
                         device, dtype):
    """
    Scheme B: caller ensures the batch is already on the SAME device/dtype as projector/model.
    We only build r_list and call the projector. This avoids extra device round-trips.
    """
    if projector is None:
        return syndromes
    B = syndromes.shape[0]
    r_list = torch.full((B,), int(r), device=device, dtype=torch.long)
    out = projector(syndromes, r_list=r_list, mask=None)
    return out

if __name__ == '__main__':
    all_data = {}
    all_dem_samplers = {}
    max_syndrome_size = 0
    max_input_size = 0

    print("Loading data for all r values...")

    # Detect D (detectors per round) from the first loaded r.
    round_pe_projector = None
    detected_D = None

    for r in r_values:
        print(f"Loading data for r = {r}")
        r_str = f"{r:02d}"
        # Paths for Google Surface Code dataset (adjust to your environment)
        path_s   = f'/ssd/userhome/maolin/qec/data/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/detection_events.b8'
        path_l   = f'/ssd/userhome/maolin/qec/data/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/obs_flips_actual.01'
        
        # [修改] 3. 使用 SHOT_TYPE 动态构建 DEM 文件名和路径
        dem_filename = f"pij_from_{SHOT_TYPE}.dem"
        path_dem = f'/ssd/userhome/maolin/qec/data/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/{dem_filename}'
      
        data = Data(d, r, path_s, path_l)
        syndromes = torch.tensor(data.syndromes()) * 1.0 # (N, D*r)
        logicals  = torch.tensor(data.logical_flip()) * 1.0

        # Detect D (per-round detector count) and initialize projector if needed
        syn_len = syndromes.size(1)
        assert syn_len % r == 0, f"syndrome_len={syn_len} not divisible by r={r}; check data layout."
        D_here = syn_len // r
        if detected_D is None:
            detected_D = D_here
        else:
            if detected_D != D_here:
                print(f"[Warn] Inconsistent D across r: seen {detected_D} vs {D_here}. Using first detected D={detected_D}.")
        if USE_ROUND_PE and round_pe_projector is None:
            round_pe_projector = RoundPositionalProjector(
                num_detectors=detected_D,
                max_rounds=ROUND_PE_MAX_ROUNDS,
                dim=ROUND_PE_DIM,
                alpha_init=ROUND_PE_ALPHA_INIT
            )

        # Build per-r input according to whether we use r_feature
        if USE_R_FEATURE:
            r_feature = torch.full((syndromes.size(0), 1), r / 10.0)
            input_data = torch.hstack((syndromes, r_feature, logicals))
        else:
            input_data = torch.hstack((syndromes, logicals))

        all_data[r] = {
            'syndromes': syndromes,   # unpadded, no-RPE yet
            'logicals':  logicals,
            'input':     input_data,
            'r_feature': torch.full((syndromes.size(0), 1), r / 10.0) if USE_R_FEATURE else None
        }

        # Prepare DEM sampler
        import stim
        dem = stim.DetectorErrorModel.from_file(path_dem)
        sampler = dem.compile_sampler()
        all_dem_samplers[r] = sampler

        # Track maxima used for padding
        max_syndrome_size = max(max_syndrome_size, syndromes.size(1))
        max_input_size    = max(max_input_size, input_data.size(1))

        print(f"r={r}: syndromes.size()={syndromes.size()}, logicals.size()={logicals.size()}")

    print(f"Max syndrome size: {max_syndrome_size}, Max input size: {max_input_size}")

    # Prepare final inputs; if not using r_feature, pad syndromes to uniform length (still on CPU)
    if USE_R_FEATURE:
        print("Preparing data with explicit r_feature; no padding required.")
        for r in r_values:
            all_data[r]['final_input'] = all_data[r]['input']
            print(f"r={r}: input size with r_feature = {all_data[r]['final_input'].size()}")
    else:
        print("Padding syndromes to a uniform length...")
        for r in r_values:
            syndromes = all_data[r]['syndromes'] # no-RPE yet
            logicals  = all_data[r]['logicals']
            pad_len = max_syndrome_size - syndromes.size(1)
            if pad_len > 0:
                padded_syndromes = torch.hstack([syndromes, torch.zeros(syndromes.size(0), pad_len)])
            else:
                padded_syndromes = syndromes
            all_data[r]['padded_syndromes'] = padded_syndromes
            all_data[r]['final_input'] = torch.hstack((padded_syndromes, logicals))
            print(f"r={r}: padded input size = {all_data[r]['final_input'].size()}")

    # ---------------- Training hyperparameters ----------------
    ni = max_input_size   # CRADLE input dimension
    epoch = 600000
    batch = 10240
    lr = 0.001
    GPU_ID = getattr(args, 'gpu_id', 0)

    if torch.cuda.is_available():
        # 检查指定的 GPU ID 是否有效
        if GPU_ID >= torch.cuda.device_count():
            print(f"警告: GPU ID {GPU_ID} 无效。系统共找到 {torch.cuda.device_count()} 块GPU。将自动使用 cuda:0。")
            device = 'cuda:0'
        else:
            device = f'cuda:{GPU_ID}'
    else:
        device = 'cpu'
    
    dtype = torch.float32
    
    print(f"*** 选定设备: {device} ***") # 增加一个明确的打印，确认设备
    # [修改] 4. 更新对 print_strategy_info 的调用，传入 SHOT_TYPE
    print_strategy_info(TRAINING_STRATEGY, CURRICULUM_WARMUP_EPOCHS, epoch, r_values, USE_R_FEATURE, SHOT_TYPE)

    if USE_ROUND_PE and round_pe_projector is not None:
        round_pe_projector = round_pe_projector.to(device).to(dtype)

    for r in r_values:
        all_data[r]['final_input'] = all_data[r]['final_input'].to(device).to(dtype)
        all_data[r]['logicals']    = all_data[r]['logicals'].to(device).to(dtype)
        all_data[r]['syndromes']   = all_data[r]['syndromes'].to(device).to(dtype)
        if not USE_R_FEATURE:
            all_data[r]['padded_syndromes'] = all_data[r]['padded_syndromes'].to(device).to(dtype)

    van = CRADLE(n=ni, depth=4, width=30, residual=False).to(device).to(dtype)
    optimizer = torch.optim.Adam(van.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)

    # ---------------- Training loop ----------------
    pbar = tqdm(range(epoch), desc="Training Progress", file=sys.stdout, dynamic_ncols=True)
    
    for l in pbar:
        r = get_r_for_epoch(l, TRAINING_STRATEGY, r_values, CURRICULUM_WARMUP_EPOCHS)
        sampler = all_dem_samplers[r]

        samples = sampler.sample(batch)
        s_syndromes = torch.tensor(samples[0]) * 1.0
        s_logicals  = torch.tensor(samples[1]) * 1.0

        s_syndromes = s_syndromes.to(device).to(dtype)
        s_logicals  = s_logicals.to(device).to(dtype)

        if USE_ROUND_PE and round_pe_projector is not None:
            s_syndromes = maybe_apply_round_pe(s_syndromes, r, round_pe_projector, device=device, dtype=dtype)

        
        s_syndromes = torch.clamp(s_syndromes, 0.0, 1.0)
        
        # [关键修改] 无论何种情况，都先对 s_syndromes 进行填充
        pad_len = max_syndrome_size - s_syndromes.size(1)
        if pad_len > 0:
            s_syndromes_padded = torch.hstack([s_syndromes, torch.zeros(batch, pad_len, device=device, dtype=dtype)])
        else:
            s_syndromes_padded = s_syndromes
        
        # 现在基于填充后的 s_syndromes_padded 来构建最终输入 s
        if USE_R_FEATURE:
            s_r_features = torch.full((batch, 1), r / 10.0, device=device, dtype=dtype)
            # 使用 s_syndromes_padded 而不是 s_syndromes
            s = torch.hstack((s_syndromes_padded, s_r_features, s_logicals))
        else:
            # 这个分支的逻辑本身是正确的，它已经使用了填充后的 s_syndromes_padded
            s = torch.hstack((s_syndromes_padded, s_logicals))

        logp = van.log_prob((s * 2 - 1))
        loss = torch.mean(-logp, dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002:
            scheduler.step()

        if (l % 1000) == 0:
            test_r = r_values[torch.randint(0, len(r_values), (1,)).item()]
            test_logicals = all_data[test_r]['logicals']
        
            # --- [关键BUG修复] ---
            # 无论 USE_R_FEATURE 是真是假，我们都需要一个被正确填充的综合征
            # 来作为模型的输入条件 (condition)
            
            # 1. 获取原始（未填充）的综合征
            raw_test_syndromes = all_data[test_r]['syndromes']
            n_test = min(50000, raw_test_syndromes.size(0))
            raw_test_syndromes = raw_test_syndromes[:n_test]
        
            # 2. 对其进行填充
            pad_len = max_syndrome_size - raw_test_syndromes.size(1)
            if pad_len > 0:
                padded_test_syndromes = torch.hstack([raw_test_syndromes, torch.zeros(n_test, pad_len, device=device, dtype=dtype)])
            else:
                padded_test_syndromes = raw_test_syndromes
                
            # 3. 根据 USE_R_FEATURE 构建最终的输入条件
            syndrome_condition_for_fwd = None
            if USE_R_FEATURE:
                r_feature_tensor = torch.full((n_test, 1), test_r / 10.0, device=device, dtype=dtype)
                syndrome_condition_for_fwd = torch.hstack((padded_test_syndromes, r_feature_tensor))
            else:
                syndrome_condition_for_fwd = padded_test_syndromes
        
            # 4. 调用 forward 函数进行评估
            lconf = forward(n_s=n_test, m=ni - 1, van=van,
                            syndrome=syndrome_condition_for_fwd,
                            device=device, dtype=dtype, k=1/2)
            # --- [BUG修复结束] ---
            
            logical_error_rate = torch.abs(test_logicals[:n_test] - lconf).sum() / n_test

            phase = "warmup" if (TRAINING_STRATEGY == 'curriculum' and l < CURRICULUM_WARMUP_EPOCHS) else "main"
            feature_mode = "w/r_feat" if USE_R_FEATURE else "no_r_feat"
            pe_mode = "pe_on" if USE_ROUND_PE else "pe_off"
            
            pbar.write(f"Epoch {l:6d} | {TRAINING_STRATEGY:10s} | {feature_mode:9s} | {pe_mode:6s} | "
                       f"{phase:6s} | Train r={r} | Test r={test_r} | Error: {logical_error_rate:.6f}", file=sys.stdout)
            
            pbar.set_postfix({'r': r, 'test_r': test_r, 'loss': f'{loss.item():.4f}'})

# ---------------- Save model ----------------
    feature_suffix = "_with_r_feat" if USE_R_FEATURE else "_no_r_feat"
    # [修改] 5. 使用 SHOT_TYPE 动态设置 shot_suffix
    shot_suffix = f"_{SHOT_TYPE}"
    pe_suffix = "_roundPE" if USE_ROUND_PE else "_noPE"
    
    # [关键修改] 在模型名称中加入 basis (Z或X) 和 d (distance)
    # 这两个变量 (basis, d) 都是在脚本顶部定义的
    model_name = f'b{basis}_d{d}_c{center}_multi_r_{TRAINING_STRATEGY}{feature_suffix}{pe_suffix}{shot_suffix}.pt'
    
    save_path = abspath(dirname(__file__)) + f'/net/cir/{model_name}'
    save_dir = dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(van, save_path)

    print("\nTraining completed!")
    # [修改] 打印更完整的信息
    print(f"Model saved to: {save_path}")
    print(f"Basis: {basis}, d={d}, center={center}")
    print(f"Strategy used: {TRAINING_STRATEGY}")
    print(f"R feature: {'ENABLED' if USE_R_FEATURE else 'DISABLED'}")
    print(f"Round-PE : {'ENABLED' if USE_ROUND_PE else 'DISABLED'}")
    # [新增] 打印 SHOT_TYPE
    print(f"Shot Type: {SHOT_TYPE.upper()}")