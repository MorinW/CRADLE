import torch
import torch.nn as nn
import sys
import os
import argparse
from tqdm import tqdm
from os.path import abspath, dirname
from typing import Optional

# --- 关键路径设置 (和训练脚本保持一致) ---
CURRENT_DIR = abspath(dirname(__file__))
QEC_DIR = dirname(CURRENT_DIR)
PARENT_DIR = dirname(QEC_DIR)

sys.path.append(QEC_DIR)
sys.path.append(PARENT_DIR)
from module import CRADLE, Data

basis, d = 'Z', 3
center = '5_3' # [修改] 必须和训练脚本的 '5_7' 保持一致
# [关键适配] 定义评估时要测试的所有 r 值
R_EVAL_VALUES = [3, 5, 7, 9, 11, 13] # [确认] 必须包含训练时的所有 r
DATA_BASE_PATH = "/ssd/userhome/maolin/qec/data"
# --- [关键适配] 核心维度，必须与你的训练脚本完全一致 ---
# [修改] 这个值必须是训练脚本中 r_values=[3...13] (d=3, center=5_7) 所产生的
# 最大的 'max_syndrome_size'。
# 如果 104 是 r=13 时的长度，那它就是对的。如果不是，请改成训练时打印的 'Max syndrome size'。
GLOBAL_MAX_SYNDROME_SIZE = 104 # [确认] 必须和训练脚本的 'Max syndrome size' 一致

# --- RPE 模块的超参数 (为了完整性保留，但当前模型未使用) ---
ROUND_PE_DIM        = 16
ROUND_PE_MAX_ROUNDS = 16 # 与训练脚本保持一致
ROUND_PE_ALPHA_INIT = 1e-2

# ------------------------------------------------------------
# Round Positional Encoding (RPE) — 从训练/评估脚本复制而来
# ------------------------------------------------------------
class RoundPositionalProjector(nn.Module):
    def __init__(self, num_detectors: int, max_rounds: int, dim: int = 16, alpha_init: float = 1e-2):
        super().__init__()
        self.num_detectors = num_detectors
        self.max_rounds = max_rounds
        self.dim = dim
        self.det_emb = nn.Embedding(num_detectors, dim)
        self.rnd_emb = nn.Embedding(max_rounds + 1, dim)
        self.proj = nn.Linear(dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        nn.init.normal_(self.det_emb.weight, std=0.02)
        nn.init.normal_(self.rnd_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)

    @torch.no_grad()
    def _build_index_maps(self, syn_len: int, r: int):
        assert r > 0 and syn_len % r == 0, f"syn_len={syn_len} must be divisible by r={r}"
        D = syn_len // r
        det_ids = torch.arange(syn_len) % D
        round_ids = torch.arange(syn_len) // D + 1
        return det_ids, round_ids, D

    def forward(self, syn_bits: torch.Tensor, r_list: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, syn_len = syn_bits.shape
        device = syn_bits.device
        dtype = syn_bits.dtype
        if mask is None:
            mask = torch.ones_like(syn_bits, dtype=dtype, device=device)
        uniq = torch.unique(r_list)
        if uniq.numel() == 1:
            r = int(uniq.item())
            det_ids, round_ids, _ = self._build_index_maps(syn_len, r)
            det_e = self.det_emb(det_ids.to(device))
            rnd_e = self.rnd_emb(round_ids.clamp_max(self.max_rounds).to(device))
            pe = self.proj(det_e + rnd_e).squeeze(-1)
            out = syn_bits + self.alpha * pe
            out = out * mask + syn_bits * (1 - mask)
            return out
        else:
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

# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------
def maybe_apply_round_pe(syndromes: torch.Tensor, r: int,
                         projector: Optional[RoundPositionalProjector],
                         device, dtype):
    if projector is None:
        return syndromes
    B = syndromes.shape[0]
    r_list = torch.full((B,), int(r), device=device, dtype=torch.long) 
    out = projector(syndromes, r_list=r_list, mask=None)
    return out

def filter_shots(syndromes, logicals, shot_type='all'):
    if shot_type == 'all':
        return syndromes, logicals
    elif shot_type == 'odd':
        odd_indices = torch.arange(1, syndromes.size(0), 2)
        return syndromes[odd_indices], logicals[odd_indices]
    elif shot_type == 'even':
        even_indices = torch.arange(0, syndromes.size(0), 2)
        return syndromes[even_indices], logicals[even_indices]
    else:
        raise ValueError("shot_type must be 'all', 'odd', or 'even'")

def forward(n_s, m, van, syndrome_condition, device, dtype, k=1):
    condition = syndrome_condition*2-1 
    x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    x = x[:, m:m+int(2*k)]
    return x

def get_detector_count(r_values_to_check):
    detected_D = None
    for r in r_values_to_check:
        r_str = f"{r:02d}"
        path_s = DATA_BASE_PATH + f'/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/detection_events.b8'
        path_l = DATA_BASE_PATH + f'/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/obs_flips_actual.01'
        try:
            data = Data(d, r, path_s, path_l)
            syndromes = torch.tensor(data.syndromes())*1.
        except Exception:
            continue
        current_syndrome_size = syndromes.size(1)
        if r > 0 and current_syndrome_size % r == 0:
            D_here = current_syndrome_size // r
            if detected_D is None:
                detected_D = D_here
                return detected_D
    return detected_D

def load_evaluation_data(r_eval_values, shot_type='all'):
    all_data = {}
    print(f"\nLoading evaluation data for r_values: {r_eval_values}...")
    
    for r in r_eval_values:
        print(f"Loading data for r = {r}")
        
        r_str = f"{r:02d}"
        base_data_path = DATA_BASE_PATH + f'/surface_code_b{basis}_d{d}_r{r_str}_center_{center}/'
        path_s = base_data_path + 'detection_events.b8'
        path_l_actual = base_data_path + 'obs_flips_actual.01'
        
        try:
            data = Data(d, r, path_s, path_l_actual)
            syndromes = torch.tensor(data.syndromes())*1.
            logicals_actual = torch.tensor(data.logical_flip(path=path_l_actual))*1.
        except Exception as e:
            print(f"ERROR: Could not load main data for r={r}. Skipping.")
            continue
        
        syndromes_filtered, logicals_filtered = filter_shots(syndromes, logicals_actual, shot_type)
            
        all_data[r] = {
            'syndromes': syndromes_filtered,
            'logicals': logicals_filtered,
        }

        baselines_raw = {}
        
        try:
            path_tn = base_data_path + 'obs_flips_predicted_by_tensor_network_contraction.01'
            baselines_raw['TN'] = torch.tensor(data.logical_flip(path=path_tn))*1.
        except Exception:
            print(f"  - Info: No Tensor Network (.01) file found for r={r}.")
        
        try:
            path_mwpm = base_data_path + 'obs_flips_predicted_by_pymatching.01'
            baselines_raw['MWPM'] = torch.tensor(data.logical_flip(path=path_mwpm))*1.
        except Exception:
            print(f"  - Info: No MWPM (pymatching) (.01) file found for r={r}.")

        try:
            path_bm = base_data_path + 'obs_flips_predicted_by_belief_matching.01'
            baselines_raw['BM'] = torch.tensor(data.logical_flip(path=path_bm))*1.
        except Exception:
            pass

        try:
            path_cm = base_data_path + 'obs_flips_predicted_by_correlated_matching.01'
            baselines_raw['CM'] = torch.tensor(data.logical_flip(path=path_cm))*1.
        except Exception:
            pass
        
        for name, tensor in baselines_raw.items():
            _, filtered_baseline = filter_shots(syndromes, tensor, shot_type)
            all_data[r][f'logicals_pre_{name}'] = filtered_baseline
        
        print(f"r={r}: loaded {syndromes_filtered.shape[0]} samples.")

    return all_data

def main(args):
    device = args.device
    dtype = torch.float32
    
    feature_suffix = "_with_r_feat" if args.r_feature else "_no_r_feat"
    # [修改] 1. 使用 args.shot_type
    # [修改] 2. 确保它生成和训练时一样的后缀 (例如 "_odd_for_even")
    shot_suffix = f"_{args.shot_type}"
    pe_suffix = "_roundPE" if args.round_pe else "_noPE"

    # [修改] 3. 添加 b{basis} 前缀，以匹配训练脚本的保存名
    # (basis 和 d 是在脚本顶部定义的全局变量)
    model_name = f'b{basis}_d{d}_c{center}_multi_r_{args.strategy}{feature_suffix}{pe_suffix}{shot_suffix}.pt'

    save_dir = abspath(dirname(__file__)) + '/net/cir/'
    model_path = save_dir + model_name
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件! 路径: {model_path}")
        sys.exit(1)

    print("\nDetermining model dimensions based on training script configuration...")
    train_max_syndrome_size = GLOBAL_MAX_SYNDROME_SIZE
    
    if args.r_feature:
        ni = train_max_syndrome_size + 1 + 1
    else:
        ni = train_max_syndrome_size + 1
    
    print(f"  Model Dims (USE_R_FEATURE={args.r_feature}):")
    print(f"  Max syndrome size from training: {train_max_syndrome_size}")
    print(f"  Calculated Model Input Dim (ni): {ni}")

    detected_D = get_detector_count(R_EVAL_VALUES)
    round_pe_projector = None
    if args.round_pe:
        if detected_D is None:
            print("错误: --round_pe=True, 但无法检测到 num_detectors (D).")
            sys.exit(1)
        round_pe_projector = RoundPositionalProjector(
            num_detectors=detected_D, max_rounds=ROUND_PE_MAX_ROUNDS,
            dim=ROUND_PE_DIM, alpha_init=ROUND_PE_ALPHA_INIT
        ).to(device).to(dtype)
        round_pe_projector.eval()

    all_data = load_evaluation_data(R_EVAL_VALUES, args.shot_filter)
    
    print(f"\nLoading model from: {model_path}")
    try:
        van = torch.load(model_path, map_location=device, weights_only=False)
        van.to(device).to(dtype)
        van.eval()
        print("Model loaded successfully.")
        
        if van.n != ni:
            print(f"!!! 维度不匹配警告 !!!")
            print(f"  模型 'van.n' (加载的): {van.n}")
            print(f"  脚本 'ni' (计算的): {ni}")
            print(f"  请确保 GLOBAL_MAX_SYNDROME_SIZE ({GLOBAL_MAX_SYNDROME_SIZE}) 与训练时一致。")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print("--- STARTING FINAL EVALUATION ---")
    print(f"Model: {model_name}")
    print(f"Evaluating r values: {R_EVAL_VALUES}")
    print("="*70)

    all_logical_error_rates = {}
    all_benchmark_lers = {}

    with torch.no_grad():
        for r in tqdm(R_EVAL_VALUES, desc="Final Evaluation Progress"):
            if r not in all_data:
                tqdm.write(f"  r = {r:2d} | No data loaded. Skipping.")
                continue
                
            test_logicals_truth = all_data[r]['logicals'].to(device).to(dtype)
            n_test = test_logicals_truth.size(0)

            raw_syndromes = all_data[r]['syndromes'].to(device).to(dtype)
            syndromes_with_pe = maybe_apply_round_pe(raw_syndromes, r, round_pe_projector, device, dtype)
            syndromes_processed = torch.clamp(syndromes_with_pe, 0.0, 1.0)
            
            syndrome_len = syndromes_processed.size(1)
            padding_needed = train_max_syndrome_size - syndrome_len
            
            if padding_needed > 0:
                pad_tensor = torch.zeros(n_test, padding_needed, device=device, dtype=dtype)
                syndrome_condition_for_fwd = torch.hstack([syndromes_processed, pad_tensor])
            elif padding_needed < 0:
                tqdm.write(f"  - Truncating r={r} data (len {syndrome_len}) to fit model size ({train_max_syndrome_size}).")
                syndrome_condition_for_fwd = syndromes_processed[:, :train_max_syndrome_size]
            else:
                syndrome_condition_for_fwd = syndromes_processed
            
            if args.r_feature:
                r_feature_tensor = torch.full((n_test, 1), r/10.0, device=device, dtype=dtype)
                syndrome_condition_for_fwd = torch.hstack((syndrome_condition_for_fwd, r_feature_tensor))

            lconf_cradle = forward(n_s=n_test, m=ni-1, van=van, syndrome_condition=syndrome_condition_for_fwd, device=device, dtype=dtype, k=1/2)
            ler_cradle = abs(test_logicals_truth - lconf_cradle).sum() / n_test
            all_logical_error_rates[r] = ler_cradle.item()
            
            benchmark_results = {}
            if 'logicals_pre_TN' in all_data[r]:
                logicals_tn = all_data[r]['logicals_pre_TN'].to(device).to(dtype)
                ler_tn = abs(test_logicals_truth - logicals_tn).sum() / n_test
                benchmark_results['TN'] = ler_tn.item()
            if 'logicals_pre_MWPM' in all_data[r]:
                logicals_mwpm = all_data[r]['logicals_pre_MWPM'].to(device).to(dtype)
                ler_mwpm = abs(test_logicals_truth - logicals_mwpm).sum() / n_test
                benchmark_results['MWPM'] = ler_mwpm.item()
            if 'logicals_pre_BM' in all_data[r]:
                logicals_bm = all_data[r]['logicals_pre_BM'].to(device).to(dtype)
                ler_bm = abs(test_logicals_truth - logicals_bm).sum() / n_test
                benchmark_results['BM'] = ler_bm.item()
            if 'logicals_pre_CM' in all_data[r]:
                logicals_cm = all_data[r]['logicals_pre_CM'].to(device).to(dtype)
                ler_cm = abs(test_logicals_truth - logicals_cm).sum() / n_test
                benchmark_results['CM'] = ler_cm.item()
            all_benchmark_lers[r] = benchmark_results
            
            tqdm_msg = f"  r = {r:2d} | CRADLE: {ler_cradle.item():.6f}"
            if 'MWPM' in benchmark_results:
                tqdm_msg += f" | MWPM: {benchmark_results.get('MWPM', float('nan')):.6f}"
            tqdm.write(tqdm_msg)

    print("\n--- FINAL EVALUATION COMPLETE ---")
    print("Summary of Logical Error Rates (LER):")
    
    header = f"  {'r':<3} | {'Samples':<7} | {'CRADLE':<10} | {'MWPM':<10} | {'TN':<10} | {'BM':<10} | {'CM':<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in R_EVAL_VALUES:
        if r in all_logical_error_rates:
            ler_cradle_str = f"{all_logical_error_rates[r]:.8f}"
            samples_str = f"{all_data[r]['logicals'].size(0):<7d}"
            bench_lers = all_benchmark_lers.get(r, {})
            ler_mwpm_str = f"{bench_lers.get('MWPM', float('nan')):.8f}"
            ler_tn_str = f"{bench_lers.get('TN', float('nan')):.8f}"
            ler_bm_str = f"{bench_lers.get('BM', float('nan')):.8f}"
            ler_cm_str = f"{bench_lers.get('CM', float('nan')):.8f}"
            print(f"  {r:<3} | {samples_str} | {ler_cradle_str:<10} | {ler_mwpm_str:<10} | {ler_tn_str:<10} | {ler_bm_str:<10} | {ler_cm_str:<10}")
        else:
            print(f"  {r:<3} | No results (data loading may have failed).")
            
    print("="*70)
    print(f"Note: Data was PADDED/TRUNCATED to fit model size (train_max_syndrome_size={train_max_syndrome_size}).")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained CRADLE QEC model.")

    parser.add_argument('--strategy', type=str, default='curriculum')
    # [修改] 使用 '--shot_type' 并匹配训练脚本的 'args.py'
    parser.add_argument('--shot_type', type=str, default='odd_for_even', 
                        choices=['odd_for_even', 'even_for_odd'],
                        help="Must match the --shot_type used during training.")

    parser.add_argument('--r_feature', action='store_true')
    parser.add_argument('--round_pe', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shot_filter', type=str, default='all', choices=['all', 'odd', 'even'])
    
    cli_args = parser.parse_args()
    main(cli_args)