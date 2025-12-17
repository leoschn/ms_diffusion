import numpy as np
# load data
from dataset.ms_dataset import ms_dataset

modelConfig = {
    'dataset_train': './data/processed_pairs_v2/train',
    'dataset_test': './data/processed_pairs_v2/test',
    "batch_size": 1,
}


# load the training data
train_data = ms_dataset(root=modelConfig["dataset_train"])
# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
# x_img = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
x_cond = np.concatenate([np.asarray(train_data[i][1]) for i in range(len(train_data))])
# print(x)
# print(x_img.shape)
# calculate the mean and std along the (0, 1) axes
# train_mean_img = np.mean(x_img, axis=(0,1,2)) #1.44
# train_std_img  = np.std(x_img, axis=(0,1,2)) #1.19
train_mean_cond = np.mean(x_cond, axis=(0, 1, 2)) #0.0008
train_std_cond  = np.std(x_cond, axis=(0, 1, 2)) #0.013
# the the mean and std