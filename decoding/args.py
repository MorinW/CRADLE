import argparse

parser = argparse.ArgumentParser()

par_common = parser.add_argument_group('common parameters')
'''para of code'''
par_common.add_argument('-c_type', type=str, default='sur',
        help='the code type of the original code, one of the labels of code, default: %(default)s')
par_common.add_argument('-n', type=int, default=72,
        help='the number of qubits, one of the labels of code')
par_common.add_argument('-d', type=int, default=5,
        help='the distance of the original code, one of the labels of code')
par_common.add_argument('-k', type=int, default=1,
        help='the number of logical qubits of the code, one of the labels of code, default: %(default)d')
par_common.add_argument('-seed', type=int, default=0,
        help='seed of random removal of stabilizers from the original code, one of the labels of code, default: %(default)d')
'''para of errors'''
par_common.add_argument('-e_model', type=str, default='dep',
        help='error model, default: %(default)d')
par_common.add_argument('-error_seed', type=int, default=51697,
        help='seed of generate errors, default: %(default)d')
par_common.add_argument('-trials', type=int, default=10000,
        help='trials of decoding, default: %(default)d')
par_common.add_argument('-er', type=float, default=0.189,
        help='the error rate for inference, default: %(default)d')


'''para of cradle'''
par_common.add_argument('-depth', type=int, default=0,
        help='depth of CRADLE, default: %(default)d')
par_common.add_argument('-width', type=int, default=1,
        help='width of CRADLE, default: %(default)d')

'''para of trade'''
par_common.add_argument('-d_model', type=int, default=256,
        help='d_model of trade, default: %(default)d')
par_common.add_argument('-n_heads', type=int, default=4,
        help='number of heads, default: %(default)d')
par_common.add_argument('-d_ff', type=int, default=256,
        help='dim of forward, default: %(default)d')
par_common.add_argument('-n_layers', type=int, default=1,
        help='number of layers, default: %(default)d')
'''para for training'''
par_common.add_argument('-n_type', type=str, default='trade', choices=['cradle', 'trade','noncradle','ncradle'],
        help='net type of training , default: %(default)s')

par_common.add_argument('-dtype', type=str, default='float32',
        choices=['float32', 'float64'],
        help='dtypes used during training, default: %(default)s')
par_common.add_argument('-device', type=str, default='cuda:0',
        help='device used during training, default: %(default)s')
par_common.add_argument('-epoch', type=int, default=10000,
        help='epoch of training, default: %(default)d')
par_common.add_argument('-batch', type=int, default=10000,
        help='batch of training, default: %(default)d')
par_common.add_argument('-lr', type=float, default=0.001,
        help='learning rate, default: %(default)d')
par_common.add_argument('-timestep', type=int, default=50,
        help='learning rate, default: %(default)d')
par_common.add_argument('-cpe', type=int, default=10000,
        help='correction per cpe epoch, default: %(default)s')

par_common.add_argument('-save', type=bool, default=False,
        help='save the results if true, default: %(default)s')

## ours
par_common.add_argument('-json_in', type=str, default=None)
par_common.add_argument('-json_out', type=str, default=None)
par_common.add_argument('-pt_path', type=str, default=None)
par_common.add_argument('-save_dir', type=str, default=None)
par_common.add_argument('-outdir', type=str, default='/home/ubuntu/xty/qecGPT-gnmld/new_figs')
par_common.add_argument('-decoder', type=str)
parser.add_argument('--resume_from', type=str, default=None, 
                    help='Path to the checkpoint file to resume training from. e.g., "net_rsur/d3pm-hybrid..._final.pt"')
parser.add_argument('--load_denoiser', type=str, default=None, 
                    help='Path to the checkpoint file to resume training from. e.g., "net_rsur/d3pm-hybrid..._final.pt"')
parser.add_argument('--api_key',type=str, default=None)
parser.add_argument('--model_name',type=str, default='gpt-4o-mini')
parser.add_argument('--provider',type=str, default='openai')
parser.add_argument('--base_url',type=str)
parser.add_argument('--use_llm', action='store_true', help='Enable LLM-assisted refinement')
parser.add_argument('--gpu_id', type=int, default=0, help='要使用的GPU ID (例如 0, 1, ...)')
parser.add_argument('--shot_type', type=str, default='odd_for_even', 
                    choices=['odd_for_even', 'even_for_odd'],
                    help='Shot type for DEM file and model name (default: %(default)s)')
parser.add_argument(
    "--use_round_pe",
    action="store_true",
    help="Enable explicit round-positional bias on syndrome bits before CRADLE."
)
parser.add_argument(
    "--round_pe_dim",
    type=int,
    default=16,
    help="Embedding dimension for detector_id/round_id (typical 16–64)."
)
parser.add_argument(
    "--round_pe_max_rounds",
    type=int,
    default=16,
    help="Maximum supported rounds r for the embedding table."
)
parser.add_argument(
    "--round_pe_alpha_init",
    type=float,
    default=1e-2,
    help="Initial scale α for the additive positional bias."
)

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
