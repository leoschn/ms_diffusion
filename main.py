import glob
import os
import pickle

import torch
from torch import optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torchvision.datasets.samplers import DistributedSampler
from tqdm import tqdm

from Diffusion import train_ms, GaussianDiffusionTrainer_ms, eval_ms
from Diffusion.model_ms import UNet
from dataset.ms_dataset import ms_dataset
from scheduler import GradualWarmupScheduler


def main(model_config = None):

    modelConfig = {
        'dataset_train': 'data/processed_pairs/train',
        'dataset_test': 'data/processed_pairs/test',
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": 1,
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
        "test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs",
        "sampledImgName": "SampledNoGuidenceImgs",
        "nrow": 8,
        "inter_eval":20
    }
    if model_config is not None:
        modelConfig = model_config

    train_ms(modelConfig)
    eval_ms(modelConfig)


if __name__ == '__main__':
    main()

    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)


    # modelConfig = {
    #     'dataset_train': 'data/processed_pairs/train',
    #     'dataset_test': 'data/processed_pairs/test',
    #     "state": "train",  # or eval
    #     "epoch": 10,
    #     "batch_size": 1,
    #     "T": 1000,
    #     "channel": 128,
    #     "channel_mult": [1, 2, 3, 4],
    #     "attn": [2],
    #     "num_res_blocks": 2,
    #     "dropout": 0.15,
    #     "lr": 1e-4,
    #     "multiplier": 2.,
    #     "beta_1": 1e-4,
    #     "beta_T": 0.02,
    #     "img_size": 32,
    #     "grad_clip": 1.,
    #     "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
    #     "training_load_weight": None,
    #     "save_weight_dir": "./Checkpoints/",
    #     "test_load_weight": "ckpt_199_.pt",
    #     "sampled_dir": "./SampledImgs/",
    #     "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    #     "sampledImgName": "SampledNoGuidenceImgs.png",
    #     "nrow": 8
    # }
    #
    # rank = int(os.environ["SLURM_PROCID"])
    # world_size = int(os.environ["SLURM_NTASKS"])
    # local_rank = int(os.environ["SLURM_LOCALID"])
    #
    # os.environ["RANK"] = str(rank)
    # os.environ["WORLD_SIZE"] = str(world_size)
    # os.environ["LOCAL_RANK"] = str(local_rank)
    #
    # nodelist = os.environ["SLURM_JOB_NODELIST"]
    # master_addr = os.popen(f"scontrol show hostname {nodelist} | head -n1").read().strip()
    # os.environ["MASTER_ADDR"] = master_addr
    # os.environ["MASTER_PORT"] = "29500"
    #
    #
    #
    # init_process_group(backend='nccl')
    # torch.cuda.set_device(local_rank)
    # scaler = GradScaler()
    # device = torch.device(f"cuda:{local_rank}")
    # dataset = ms_dataset(root=modelConfig["dataset_train"])
    # sampler=DistributedSampler(dataset,shuffle=True)
    # dataloader = DataLoader(
    #     dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=3, drop_last=True, pin_memory=True,sampler=sampler)
    #
    # net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
    #                  attn=modelConfig["attn"],
    #                  num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    # net_model.apply(lambda m: setattr(m, 'weight', m.weight.contiguous())
    # if hasattr(m, 'weight') else None)
    # net_model = torch.nn.parallel.DistributedDataParallel(net_model,device_ids=[local_rank],
    # output_device=local_rank)
    #
    #
    #
    # if modelConfig["training_load_weight"] is not None:
    #     net_model.module.load_state_dict(torch.load(os.path.join(
    #         modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    # optimizer = torch.optim.AdamW(
    #     net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    # warmUpScheduler = GradualWarmupScheduler(
    #     optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
    #     after_scheduler=cosineScheduler)
    # trainer = GaussianDiffusionTrainer_ms(
    #     net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    #
    # # start training
    # for e in range(modelConfig["epoch"]):
    #     sampler.set_epoch(e)
    #     if rank == 0:
    #         pbar = tqdm(dataloader, dynamic_ncols=True)
    #     else :
    #         pbar = dataloader
    #     for images, cond in pbar:
    #         # train
    #         optimizer.zero_grad()
    #         with autocast(device_type='cuda', dtype=torch.float16):
    #             cond = cond.float().to(device)
    #             x_0 = images.float().to(device)
    #
    #             loss = trainer(x_0, cond).sum() / 1000.
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(
    #             net_model.parameters(), modelConfig["grad_clip"])
    #         scaler.step(optimizer)
    #         scaler.update()
    #         if rank == 0:
    #             lr = warmUpScheduler.get_lr()[0]
    #             pbar.set_postfix(ordered_dict={
    #                 "epoch": e,
    #                 "loss: ": loss.item(),
    #                 "img shape: ": x_0.shape,
    #                 "LR": lr
    #             })
    #     warmUpScheduler.step()
    #
    # if rank == 0:
    #     torch.save(net_model.module.state_dict(), os.path.join(
    #         modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
    #
    # destroy_process_group()