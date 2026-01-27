import os
import pickle
from typing import Dict
import wandb
import torch
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets.samplers import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from wandb.cli.cli import offline

import Diffusion
from Diffusion import GaussianDiffusionSampler_ms, GaussianDiffusionTrainer_ms
from dataset.ms_dataset import ms_dataset
from scheduler import GradualWarmupScheduler

def train_ms(modelConfig: Dict):


    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

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
    scaler = GradScaler()
    device = torch.device(f"cuda:{local_rank}")
    #train data
    dataset_train = ms_dataset(root=modelConfig["dataset_train"],im_size=modelConfig["im_size"])
    sampler_train=DistributedSampler(dataset_train,shuffle=True)
    dataloader_train = DataLoader(
        dataset_train, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=3, drop_last=True, pin_memory=True,sampler=sampler_train)

    #test data
    dataset_test = ms_dataset(root=modelConfig["dataset_test"],im_size=modelConfig["im_size"])
    sampler_test = DistributedSampler(dataset_test, shuffle=True)
    sampler_test.set_epoch(0)
    dataloader_test = DataLoader(
        dataset_test, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=3, drop_last=True,
        pin_memory=True, sampler=sampler_test)

    #model

    if modelConfig["model"] == 'v1':
        net_model = Diffusion.model_ms.UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                         attn=modelConfig["attn"],
                         num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                         window_embedding=modelConfig["window_embd"], n_window=modelConfig["n_window"]).to(device)
    elif modelConfig["model"]=='v2':
        net_model = Diffusion.model_ms_v2.UNet(T=modelConfig["T"], ch=modelConfig["channel"],
                                            ch_mult=modelConfig["channel_mult"],
                                            attn=modelConfig["attn"],
                                            num_res_blocks=modelConfig["num_res_blocks"],
                                            dropout=modelConfig["dropout"],
                                            window_embedding=modelConfig["window_embd"],
                                            n_window=modelConfig["n_window"]).to(device)
    else:
        raise 'model not found'
    net_model.apply(lambda m: setattr(m, 'weight', m.weight.contiguous())
    if hasattr(m, 'weight') else None)
    net_model = torch.nn.parallel.DistributedDataParallel(net_model,device_ids=[local_rank],
    output_device=local_rank)

    #wandb init
    if rank == 0:
        with open('wdb_key.txt', 'r') as f:
            key = f.readline().strip()
        os.environ["WANDB_API_KEY"] = key

        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="liris",
            # Set the wandb project where this run will be logged.
            project="ms diffusion",
            # Track hyperparameters and run metadata.
            config=modelConfig,
            mode="offline",
            name="train_ms_1",
            dir = './wandb_run'
        )

    if modelConfig["training_load_weight"] is not None:
        net_model.module.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["warmup_epoches"],
        after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer_ms(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    sampler = GaussianDiffusionSampler_ms(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    # Sampled from standard normal distribution
    mse_loss_fct = torch.nn.MSELoss()

    # start training
    for e in range(modelConfig["epoch"]):
        total_loss = 0
        sampler_train.set_epoch(e)
        if rank == 0:

            i = 0
            pbar = tqdm(dataloader_train, dynamic_ncols=True)
        else :
            pbar = dataloader_train
        for images, cond, _, wind in pbar:

            # train
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                cond = cond.to(device, non_blocking=True)
                x_0 = images.to(device, non_blocking=True)
                wind = wind.to(device, non_blocking=True)

                loss = trainer(x_0, cond, wind).sum()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                i += 1
                lr = warmUpScheduler.get_lr()[0]
                pbar.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": total_loss/i,
                    "img shape: ": x_0.shape,
                    "LR": lr
                })
        if rank == 0:
            run.log({"epoch":e,"loss": total_loss/i,"LR": lr})
        warmUpScheduler.step()

        if rank == 0:
            torch.save(net_model.module.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

        if e%modelConfig["inter_eval"]==0:
            net_model.eval()

            with torch.no_grad():
                n_image = 0
                if rank == 0:
                    pbar_test =  tqdm(dataloader_test, dynamic_ncols=True)
                else :
                    pbar_test = dataloader_test
                total_mse = 0
                total_psnr = 0
                for images, cond ,path ,wind in pbar_test:
                    n_image+=1
                    f_name = os.path.basename(path[0]).replace('.pkl', '')



                    with autocast(device_type='cuda', dtype=torch.float16,enabled=True):
                        cond = cond.to(device)
                        images = images.to(device)
                        wind = wind.int().to(device)
                        noisyImage = torch.randn_like(images[:, :1, :, :])
                        sampledImgs = sampler(noisyImage, cond, wind)
                    mse_loss = mse_loss_fct(sampledImgs, images)
                    psnr_loss = 10 * torch.log10(1 / mse_loss)


                    total_mse += mse_loss.item()
                    total_psnr += psnr_loss.item()


                    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
                    arr = sampledImgs.cpu().numpy()
                    with open( os.path.join(
                        modelConfig["sampled_dir"], f_name + '_' + str(e) + '.pkl'),'wb') as f:
                        pickle.dump(arr, f)
                    save_image(sampledImgs, os.path.join(
                        modelConfig["sampled_dir"], f_name + '_' + str(e) + '.png'),
                               nrow=modelConfig["nrow"])
                    if rank == 0:
                        run.log({'sampled image': wandb.Image(os.path.join(
                        modelConfig["sampled_dir"], f_name + '_' + str(e) + '.png'))})
                print(f"mse loss gpu {rank} epoch {e}: ", total_mse / n_image)
                print(f"psnr loss gpu {rank} epoch {e}:", total_psnr / n_image)
                if rank == 0:
                    run.log({
                        "epoch_eval": e,
                        "loss_eval": total_mse / n_image,
                        "psnr_eval": total_psnr / n_image
                    })

    if rank == 0:
        run.finish()
    destroy_process_group()
