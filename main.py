from Diffusion import train_ms, eval_ms

def main(model_config = None):

    modelConfig = {
        'dataset_train': 'data/processed_pairs/train',
        'dataset_test': 'data/processed_pairs/test',
        "state": "train",  # or eval
        "epoch": 20, #including 10 warming epoch (error if total < warming)
        "batch_size": 1,
        "n_window":0,
        "window_embd":False,
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
        "inter_eval":1
    }
    if model_config is not None:
        modelConfig = model_config

    train_ms(modelConfig)
    # eval_ms(modelConfig)


if __name__ == '__main__':
    main()
