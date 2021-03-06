import pytorch_lightning as pl
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class DataModulePL(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=2, num_workers=4, split_ratio=0.8):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        train_dataset, val_dataset = train_test_split(dataset, test_size=1.0 - split_ratio, random_state=0)

        self.train_dataset = DataSetMap(train_dataset)
        self.val_dataset = DataSetMap(val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class DataSetMap(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]

        return self.data[idx]

    def __len__(self):
        return len(self.data)
