import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--warmup_epoches', type=int, default=1)
    parser.add_argument('--save_inter', type=int, default=50)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--n_window', type=int, default=164)
    parser.add_argument('--window_type', type=str, default='all')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--window_embd', type=str, default='categorical')
    parser.add_argument('--save_dir', type=str, default='./SampledImgs_2/')
    parser.add_argument('--attn', type=int, default=4)
    parser.add_argument('--amp', type=str, default='f16')
    parser.add_argument('--model', type=str, default='v2')
    parser.add_argument('--schema', type=str, default='FSDP2')
    args = parser.parse_args()

    return args



