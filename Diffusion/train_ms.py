import datetime
import os
import pickle
from typing import Dict

import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import Diffusion
from dataset.ms_dataset import ms_dataset
from scheduler import GradualWarmupScheduler


def train_ms(modelConfig: Dict):

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    print(rank, local_rank, world_size)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    master_addr = os.popen(f"scontrol show hostname {nodelist} | head -n1").read().strip()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"

    init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)  # Extend from 10 to 60 minutes
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    #train data
    dataset_train = ms_dataset(root=modelConfig["dataset_train"],im_size=modelConfig["im_size"],window=modelConfig["dataset_window"],total_windows=modelConfig["n_window"])
    sampler_train=DistributedSampler(dataset_train,shuffle=True,drop_last=False)
    dataloader_train = DataLoader(
        dataset_train, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=1, drop_last=False, pin_memory=True,sampler=sampler_train)

    dataset_val = ms_dataset(root=modelConfig["dataset_val"],im_size=modelConfig["im_size"],window=modelConfig["dataset_window"],total_windows=modelConfig["n_window"])
    sampler_val=DistributedSampler(dataset_val,shuffle=False,drop_last=True)
    dataloader_val = DataLoader(
        dataset_val, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=1, drop_last=False, pin_memory=True,sampler=sampler_val)


    #test data
    dataset_test = ms_dataset(root=modelConfig["dataset_test"],im_size=modelConfig["im_size"],window=modelConfig["dataset_window"],total_windows=modelConfig["n_window"])
    sampler_test = DistributedSampler(dataset_test, shuffle=False, drop_last=True)
    sampler_test.set_epoch(0)
    dataloader_test = DataLoader(
        dataset_test, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=1, drop_last=True,
        pin_memory=True, sampler=sampler_test)

    #model

    if modelConfig["model"] == 'v1':
        net_model = Diffusion.model_ms.UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                         attn=modelConfig["attn"],
                         num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                         window_embedding=modelConfig["window_embd"], n_window=modelConfig["n_window"]).to(device)
    elif modelConfig["model"]=='add':
        net_model = Diffusion.model_ms_v2.UNet(T=modelConfig["T"], ch=modelConfig["channel"],
                                            ch_mult=modelConfig["channel_mult"],
                                            attn=modelConfig["attn"],
                                            num_res_blocks=modelConfig["num_res_blocks"],
                                            dropout=modelConfig["dropout"],
                                            window_embedding=modelConfig["window_embd"],
                                            n_window=modelConfig["n_window"]).to(device)
    elif modelConfig["model"]=='concat':
        net_model = Diffusion.model_ms_concat.UNet(T=modelConfig["T"], ch=modelConfig["channel"],
                                            ch_mult=modelConfig["channel_mult"],
                                            attn=modelConfig["attn"],
                                            num_res_blocks=modelConfig["num_res_blocks"],
                                            dropout=modelConfig["dropout"],
                                            window_embedding=modelConfig["window_embd"],
                                            n_window=modelConfig["n_window"]).to(device)
    else:
        raise ValueError("model type not found")
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
    if modelConfig['type']=='ddpm':

        if modelConfig['thresholding'] == 'fix':
            trainer = Diffusion.diffusion_ms.GaussianDiffusionTrainer_ms(
                net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            sampler = Diffusion.diffusion_ms.GaussianDiffusionSampler_ms(
                net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        elif modelConfig['thresholding'] == 'dyn':
            trainer = Diffusion.diffusion_ms_dyn.GaussianDiffusionTrainer_ms(
                net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],modelConfig["loss"]).to(device)
            sampler = Diffusion.diffusion_ms_dyn.GaussianDiffusionSampler_ms(
                net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],modelConfig["num_threshold"]).to(device)

        else :
            raise NotImplementedError

    elif modelConfig['type']=='ddim':
        trainer = Diffusion.diffusion_DDIM.GaussianDiffusionTrainer_ms(
            net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["loss"]).to(device)
        sampler = Diffusion.diffusion_DDIM.GaussianDiffusionSampler_ms(
            net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],modelConfig["num_threshold"],
            eta=modelConfig["eta"],ddim_steps=modelConfig["ddim_steps"]).to(device)
    else :
        raise NotImplementedError

    # Sampled from standard normal distribution
    mse_loss_fct = torch.nn.MSELoss()

    # start training
    best_epoch = 0
    best_loss = np.inf
    for e in range(modelConfig["epoch"]):
        net_model.train()
        total_loss = 0
        n_image=0
        sampler_train.set_epoch(e)
        if rank == 0:
            pbar = tqdm(dataloader_train, dynamic_ncols=True)
        else :
            pbar = dataloader_train
        for images, cond, _, wind in pbar:
            # train
            batch_size = images.size(0) * world_size
            n_image += batch_size
            optimizer.zero_grad(set_to_none=True)

            cond = cond.to(device, non_blocking=True)
            x_0 = images.to(device, non_blocking=True)
            # print('cond : ' ,cond.min(), cond.max(), cond.mean(), cond.std())
            wind = wind.to(device, non_blocking=True)

            loss = trainer(x_0, cond, wind).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"]
            )
            optimizer.step()

            loss_detached = loss.detach()

            torch.distributed.all_reduce(
                loss_detached,
                op=torch.distributed.ReduceOp.SUM
            )

            loss_detached /= world_size

            if rank == 0:
                total_loss+=loss_detached.item()*batch_size
                lr = warmUpScheduler.get_lr()[0]
                pbar.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": total_loss/n_image,
                    "img shape: ": x_0.size(),
                    "LR": lr
                })
        if rank == 0:
            run.log({"epoch":e,"loss": total_loss/n_image,"LR": lr})
        warmUpScheduler.step()

        if rank == 0:
            os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
            torch.save(net_model.module.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        torch.distributed.barrier()


        if e%modelConfig["inter_eval"]==modelConfig["inter_eval"]-1:
            net_model.eval()
            sampler_val.set_epoch(e)

            with torch.no_grad():
                total_loss=0
                n_image = 0
                if rank == 0:
                    pbar_val =  tqdm(dataloader_val, dynamic_ncols=True)
                else :
                    pbar_val = dataloader_val
                total_mse = 0
                total_psnr = 0
                for images, cond ,path ,wind in pbar_val:
                    batch_size = images.size(0) * world_size
                    n_image+=batch_size
                    cond = cond.to(device, non_blocking=True)
                    images = images.to(device, non_blocking=True)
                    wind = wind.int().to(device, non_blocking=True)
                    noisyImage = torch.randn_like(images[:, :1, :, :])
                    sampledImgs = sampler(noisyImage, cond, wind)
                    mse_loss = mse_loss_fct(sampledImgs, images)
                    psnr_loss = 10 * torch.log10(4 / mse_loss) #image in [-1 1]


                    total_mse += mse_loss.item() *batch_size
                    total_psnr += psnr_loss.item() *batch_size


                    if rank == 0:
                        lr = warmUpScheduler.get_lr()[0]
                        pbar_val.set_postfix(ordered_dict={
                            "epoch": e,
                            "loss: ": total_mse/n_image,
                            "img shape: ": images.size(),
                            "LR": lr
                        })

                    if rank == 0:
                        os.makedirs(modelConfig["sampled_dir_val"], exist_ok=True)

                        sampled_cpu = sampledImgs.cpu()

                        for i in range(images.size(0)):
                            fname = os.path.basename(path[i]).replace('.pkl', '')

                            # save pkl (single image)
                            arr = sampled_cpu[i].numpy()
                            with open(
                                    os.path.join(modelConfig["sampled_dir_val"], f"{fname}_{e}.pkl"),
                                    'wb'
                            ) as f:
                                pickle.dump(arr, f)

                            # save png (single image)
                            save_path = os.path.join(
                                modelConfig["sampled_dir_val"], f"{fname}_{e}.png"
                            )
                            save_image(sampled_cpu[i], save_path)

                            # optional: log only first few to wandb to avoid spam
                            if i == 0:
                                run.log({'sampled image': wandb.Image(save_path)})

                torch.distributed.barrier()

                total_mse_tensor = torch.tensor(total_mse, device=device)
                total_psnr_tensor = torch.tensor(total_psnr, device=device)
                total_count_tensor = torch.tensor(n_image, device=device)

                torch.distributed.all_reduce(total_mse_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_psnr_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_count_tensor, op=torch.distributed.ReduceOp.SUM)

                mean_mse = total_mse_tensor / total_count_tensor
                if mean_mse < best_loss:
                    best_loss = mean_mse
                    best_epoch = e
                mean_psnr = total_psnr_tensor / total_count_tensor

                if rank == 0:
                    print(f"[Eval] epoch {e} | mse: {mean_mse.item():.6f} | psnr: {mean_psnr.item():.4f}")
                    run.log({
                        "epoch_eval": e,
                        "loss_eval": mean_mse.item(),
                        "psnr_eval": mean_psnr.item(),
                    })

            torch.distributed.barrier()

    #apply best model on test set

    ckpt = torch.load(
        os.path.join(
            modelConfig["save_weight_dir"],
            f'ckpt_{best_epoch}_.pt'
        ),
        map_location=device
    )

    net_model.module.load_state_dict(ckpt)

    net_model.eval()
    sampler_test.set_epoch(best_epoch)

    with torch.no_grad():
        n_image = 0
        if rank == 0:
            pbar_test = tqdm(dataloader_test, dynamic_ncols=True)
        else:
            pbar_test = dataloader_test
        total_mse = 0
        total_psnr = 0
        for images, cond, path, wind in pbar_test:
            batch_size = images.size(0)
            n_image += batch_size

            cond = cond.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            wind = wind.int().to(device, non_blocking=True)
            noisyImage = torch.randn_like(images[:, :1, :, :])
            sampledImgs = sampler(noisyImage, cond, wind)
            mse_loss = mse_loss_fct(sampledImgs, images)
            psnr_loss = 10 * torch.log10(1 / mse_loss)

            total_mse += mse_loss.item() * batch_size
            total_psnr += psnr_loss.item() * batch_size

            os.makedirs(modelConfig["sampled_dir_test"], exist_ok=True)

            sampled_cpu = sampledImgs.cpu()

            for i in range(images.size(0)):
                fname = os.path.basename(path[i]).replace('.pkl', '')

                # save pkl (single image)
                arr = sampled_cpu[i].numpy()
                with open(
                        os.path.join(modelConfig["sampled_dir_test"], f"{fname}_{e}.pkl"),
                        'wb'
                ) as f:
                    pickle.dump(arr, f)

                # save png (single image)
                save_path = os.path.join(
                    modelConfig["sampled_dir_test"], f"{fname}_{e}.png"
                )
                save_image(sampled_cpu[i], save_path)

        torch.distributed.barrier()

        total_mse_tensor = torch.tensor(total_mse, device=device)
        total_psnr_tensor = torch.tensor(total_psnr, device=device)
        total_count_tensor = torch.tensor(n_image, device=device)

        torch.distributed.all_reduce(total_mse_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_psnr_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_count_tensor, op=torch.distributed.ReduceOp.SUM)

        mean_mse = total_mse_tensor / total_count_tensor
        mean_psnr = total_psnr_tensor / total_count_tensor

        if rank == 0:
            print(f"[Test] epoch {best_epoch} | mse: {mean_mse.item():.6f} | psnr: {mean_psnr.item():.4f}")
            run.log({
                "epoch_test": best_epoch,
                "loss_test": mean_mse.item(),
                "psnr_test": mean_psnr.item(),
            })
    torch.distributed.barrier()

    if rank == 0:
        run.finish()
    destroy_process_group()
