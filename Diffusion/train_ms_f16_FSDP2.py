import os
import pickle
from typing import Dict
import wandb
import torch
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

import Diffusion
from Diffusion import GaussianDiffusionTrainer_ms, GaussianDiffusionSampler_ms
from dataset.ms_dataset import ms_dataset
from scheduler import GradualWarmupScheduler

def train_ms(modelConfig: Dict):

    # -------- SLURM / process setup --------
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    master_addr = os.popen(
        f"scontrol show hostname {nodelist} | head -n1"
    ).read().strip()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # -------- Data --------
    dataset_train = ms_dataset(
        root=modelConfig["dataset_train"],
        im_size=modelConfig["im_size"]
    )
    sampler_train = DistributedSampler(dataset_train, shuffle=True)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=modelConfig["batch_size"],
        sampler=sampler_train,
        num_workers=3,
        pin_memory=True,
        drop_last=True,
    )

    dataset_test = ms_dataset(
        root=modelConfig["dataset_test"],
        im_size=modelConfig["im_size"]
    )
    sampler_test = DistributedSampler(dataset_test, shuffle=True)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=modelConfig["batch_size"],
        sampler=sampler_test,
        num_workers=3,
        pin_memory=True,
        drop_last=True,
    )

    # -------- Model --------
    net_model = Diffusion.model_ms_v2.UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"],
        window_embedding=modelConfig["window_embd"],
        n_window=modelConfig["n_window"],
    ).to(device)

    #Avoid issues with frozen zero linear layers
    for p in net_model.parameters():
        p.requires_grad = True

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000,
    )

    net_model = FSDP(
        net_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )

    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4,
    )

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0,
    )

    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["warmup_epoches"],
        after_scheduler=cosineScheduler,
    )

    scaler = GradScaler()

    trainer = GaussianDiffusionTrainer_ms(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
    ).to(device)

    sampler = GaussianDiffusionSampler_ms(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
    ).to(device)

    # -------- Wandb --------
    if rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(
            project="ms diffusion",
            config=modelConfig,
            name="train_eval_fsdpA",
            dir="./wandb_run",
        )

    # -------- Training Loop --------
    for epoch in range(modelConfig["epoch"]):
        sampler_train.set_epoch(epoch)
        net_model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader_train, disable=(rank != 0))
        for i, (images, cond, _, wind) in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", torch.float16):
                images = images.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                wind = wind.to(device, non_blocking=True)
                loss = trainer(images, cond, wind).sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if rank == 0:
                pbar.set_postfix(loss=total_loss / (i + 1),
                                 lr=warmUpScheduler.get_lr()[0])

        warmUpScheduler.step()

        # -------- Save Checkpoint --------
        if rank == 0:
            with FSDP.state_dict_type(
                net_model, StateDictType.FULL_STATE_DICT
            ):
                cpu_state = net_model.state_dict()
            torch.save(
                cpu_state,
                os.path.join(
                    modelConfig["save_weight_dir"],
                    f"ckpt_{epoch}.pt"
                ),
            )
            if rank == 0:
                run.log({"epoch": epoch,"train_loss": total_loss / (i + 1),"lr": warmUpScheduler.get_lr()[0],})

        # -------- Evaluation (FSDP) --------
        net_model.eval()
        total_mse, total_psnr, n_image = 0.0, 0.0, 0
        for images, cond, path, wind in dataloader_test:
            n_image += images.size(0)
            images = images.to(device)
            cond = cond.to(device)
            wind = wind.to(device)

            with autocast("cuda", torch.float16):
                noise = torch.randn_like(images)
                sampled = sampler(noise, cond, wind)

            mse = torch.nn.functional.mse_loss(sampled, images)
            psnr = 10 * torch.log10(1 / mse)

            total_mse += mse.item() * images.size(0)
            total_psnr += psnr.item() * images.size(0)

        if rank == 0:
            run.log({"epoch_eval": epoch,"mse_eval": total_mse / n_image,"psnr_eval": total_psnr / n_image,})

    if rank == 0:
        run.finish()
    destroy_process_group()
