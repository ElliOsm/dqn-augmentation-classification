from torch.utils.data import Dataset
import torch


class DataloaderSubsetDataset(Dataset):

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)

        # if this line is commented it will rotate the image by 90 degrees
        x = x.permute((0, 2, 1))

        return x, y
