import os
from typing import Dict

import torch
import torch.optim as optim
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets.samplers import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler_ms, GaussianDiffusionTrainer_ms
from Diffusion import UNet
from dataset.ms_dataset import ms_dataset
from scheduler import GradualWarmupScheduler

def train_ms(modelConfig: Dict):

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    master_addr = os.popen(f"scontrol show hostname {nodelist} | head -n1").read().strip()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"



    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    device = torch.device(modelConfig["device"])
    dataset = ms_dataset(root=modelConfig["dataset"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True,sampler=DistributedSampler(dataset,shuffle=True))

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    net_model = torch.nn.parallel.DataParallel(net_model)


    if modelConfig["training_load_weight"] is not None:
        net_model.module.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.module.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer_ms(
        net_model.module, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        if rank == 0:
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for images, cond in tqdmDataLoader:
                    # train
                    optimizer.zero_grad()
                    cond = cond.float().to(device)
                    x_0 = images.float().to(device)

                    loss = trainer(x_0, cond).sum() / 1000.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net_model.module.parameters(), modelConfig["grad_clip"])
                    optimizer.step()
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })
            warmUpScheduler.step()
            torch.save(net_model.module.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        else :
            for images, cond in dataloader:
                # train
                optimizer.zero_grad()
                cond = cond.float().to(device)
                x_0 = images.float().to(device)

                loss = trainer(x_0, cond).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.module.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

    destroy_process_group()


def eval_ms(modelConfig: Dict):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    master_addr = os.popen(f"scontrol show hostname {nodelist} | head -n1").read().strip()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # eval dataset loading
    dataset = ms_dataset(root=modelConfig["dataset_val"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True,sampler=DistributedSampler(dataset,shuffle=False))



    # load Diffusion and evaluate
    with torch.no_grad():

        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("Diffusion load weight done.")
        model.eval()

        sampler = GaussianDiffusionSampler_ms(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        mse_loss_fct = torch.nn.MSELoss()

        if rank == 0:
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                n_image=0
                for images, cond in tqdmDataLoader:
                    total_mse=0
                    total_psnr=0
                    noisyImage = torch.randn(
                        size=[modelConfig["batch_size"], 1, 512, 2056], device=device)
                    sampledImgs = sampler(noisyImage, cond)

                    mse_loss = mse_loss_fct(sampledImgs, images)
                    psnr_loss = 10 * torch.log10(1/mse_loss)

                    total_mse += mse_loss.item()
                    total_psnr += psnr_loss.item()


                    save_image(sampledImgs, os.path.join(
                        modelConfig["sampled_dir"],  modelConfig["sampledImgName"]+str(rank)+'_'+str(n_image)), nrow=modelConfig["nrow"])
                    n_image += 1
        else :
            n_image = 0
            for images, cond in dataloader:
                total_mse = 0
                total_psnr = 0
                noisyImage = torch.randn(
                    size=[modelConfig["batch_size"], 1, 512, 2056], device=device)
                sampledImgs = sampler(noisyImage, cond)

                mse_loss = mse_loss_fct(sampledImgs, images)
                psnr_loss = 10 * torch.log10(1 / mse_loss)

                total_mse += mse_loss.item()
                total_psnr += psnr_loss.item()

                save_image(sampledImgs, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledImgName"] + str(rank) + '_' + str(n_image)),
                           nrow=modelConfig["nrow"])
                n_image += 1
                print(f"mse loss gpe {rank}: ", total_mse/n_image)
                print(f"psnr loss: {rank}", total_psnr/n_image)

    destroy_process_group()