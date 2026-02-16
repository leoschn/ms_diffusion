import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ms_dataset import ms_dataset


def compute_mean_std(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    n_channels = None
    total_pixels = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0
    max_int = 0.0
    min_int = 100000.
    for images, _, _ ,_ in tqdm(loader):
        # images: (B, C, H, W)

        if n_channels is None:
            n_channels = images.size(1)
            channel_sum = torch.zeros(n_channels)
            channel_sum_sq = torch.zeros(n_channels)

        b, c, h, w = images.shape
        pixels = b * h * w
        total_pixels += pixels
        if max_int < torch.max(images):
            max_int = torch.max(images)
        if min_int > torch.min(images):
            min_int = torch.min(images)

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sum_sq / total_pixels - mean ** 2)

    return mean, std, max_int, min_int


dataset_train = ms_dataset(root='/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/train',im_size=(256,512),window='all')

mean, std,max_int,min_int = compute_mean_std(dataset_train)
print("Mean:", mean)
print("Std:", std)
print("Max int:", max_int)
print("Min int:", min_int)