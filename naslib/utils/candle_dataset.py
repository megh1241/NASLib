import os.path as osp
import torch
from skimage import io
from torch.utils.data import Dataset
import h5py
from . import load_ops



class CandleDataset(Dataset):
    def __init__(self, data_dir, subset='train', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        h5f = h5py.File(osp.join(self.data_dir,'training_attn.h5'), "r")
        X_key = 'X_'+subset
        y_key = 'Y_'+subset
        self.X = h5f[X_key][:]
        self.Y = h5f[y_key][:]
        h5f.close()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


def get_candle_dataset(data_dir):
    train_data = CandleDataset(data_dir, subset='train')
    test_data = CandleDataset(data_dir, subset='test')
    val_data = CandleDataset(data_dir, subset='val')
    return train_data, val_data, test_data
