import pytorch_lightning
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class DataModulePL(pytorch_lightning.LightningDataModule):
    def __init__(self, dataset, batch_size=2, num_workers=4):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=0)

        self.train_dataset = DataSetMap(train_dataset)
        self.val_dataset = DataSetMap(val_dataset)

    def collate(self, x):
        return x

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate,
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
