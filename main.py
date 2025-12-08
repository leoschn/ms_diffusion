from Diffusion import train_ms, eval_ms
from config import load_args

def main(model_config = None):
    args = load_args()
    modelConfig = {
        'dataset_train': 'data/processed_pairs_v2/train',
        'dataset_test': 'data/processed_pairs_v2/test',
        "state": "train",  # or eval
        "epoch": args.epoches, #including 10 warming epoch (error if total < warming)
        "batch_size": 1,
        "n_window":args.n_window,
        "window_embd":args.window_embd,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs_2/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs",
        "sampledImgName": "sampled_ms_image",
        "nrow": 8,
        "inter_eval":20
    }
    if model_config is not None:
        modelConfig = model_config

    train_ms(modelConfig)
    # eval_ms(modelConfig)


if __name__ == '__main__':
    main()
