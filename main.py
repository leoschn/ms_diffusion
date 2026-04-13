from Diffusion import train_ms
from config import load_args

def main(model_config = None):
    args = load_args()
    modelConfig = {
        'dataset_train': '/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/train',
        'dataset_test': '/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/test',
        'dataset_window':args.window_type,
        "state": "train",  # or eval
        "epoch": args.epoches,
        "warmup_epoches": max(args.warmup_epoches,args.epoches//10),
        "batch_size": args.batch_size,
        "n_window":args.n_window,
        "window_embd":args.window_embd,
        "T": 1000,
        "im_size": (256,512),
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": args.attn,
        "loss":args.loss,
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.,
        "training_load_weight": None,
        "save_weight_dir": args.checkpoint,
        "sampled_dir": args.save_dir,
        "sampledNoisyImgName": "NoisyNoGuidenceImgs",
        "sampledImgName": "sampled_ms_image",
        "nrow": 8,
        "inter_eval":args.eval_inter,
        "model":args.model,
        "thresholding":args.thresholding,
        "num_threshold":args.num_threshold,
        "eta":args.eta,
        "ddim_steps":args.ddim_steps,
        "type":args.type,
    }
    if model_config is not None:
        modelConfig = model_config

    train_ms(modelConfig)

    # eval_ms(modelConfig)


if __name__ == '__main__':
    main()
