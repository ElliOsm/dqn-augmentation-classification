from torch.utils.data import Dataset
import pandas as pd

class DataloaderSubsetDataset(Dataset):

    def __init__(self, subset, dir, transform=None):
        self.subset = subset
        self.img_names = pd.read_csv(dir + '/train.csv')
        self.transform = transform
        self.img_dir = dir + '/train.csv'

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]

        y = self.img_names.iloc[idx, 1]


        # for i in range(len(self.img_names)):
        #     if x == self.img_names["fileName"] :
        #         y = self.img_names.iloc[idx, 1]

        if self.transform:
            x = self.transform(x)

        return x, y