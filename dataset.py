import torch
from torch.utils.data import Dataset

class SpO2Dataset(Dataset):
    def __init__(self, sequence, window_size=60):
        self.sequence = sequence
        self.window_size = window_size
        self.samples = self.create_samples()

    def create_samples(self):
        X = []
        for i in range(len(self.sequence) - self.window_size):
            segment = self.sequence[i : i + self.window_size]
            X.append(segment)
        return torch.stack(X)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]
