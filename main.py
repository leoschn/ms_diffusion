from Diffusion import train_ms_bf16, eval_ms, train_ms_f16, train_ms_f16_FSDP2
from config import load_args

def main(model_config = None):
    args = load_args()
    modelConfig = {
        'dataset_train': '/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/train',
        'dataset_test': '/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/test',
        "state": "train",  # or eval
        "epoch": args.epoches,
        "warmup_epoches": args.warmup_epoches,
        "batch_size": 1,
        "n_window":args.n_window,
        "window_embd":args.window_embd,
        "T": 1000,
        "im_size": (256,512),
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [args.attn] if args.attn < 4 else [],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": args.save_dir,
        "sampledNoisyImgName": "NoisyNoGuidenceImgs",
        "sampledImgName": "sampled_ms_image",
        "nrow": 8,
        "inter_eval":args.eval_inter,
        "amp":args.amp,
        "model":args.model,
        "schema":args.schema,
    }
    if model_config is not None:
        modelConfig = model_config

    if modelConfig["amp"]=='f16':
        if modelConfig["schema"] == 'DDP':
            train_ms_f16.train_ms(modelConfig)
        elif modelConfig["schema"] == 'FSDP2':
            train_ms_f16_FSDP2.train_ms(modelConfig)
    else :
        train_ms_bf16.train_ms(modelConfig)

    # eval_ms(modelConfig)


if __name__ == '__main__':
    main()
