import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoches', type=int, default=3)
    parser.add_argument('--save_inter', type=int, default=50)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--n_window', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--window_embd', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    return args



