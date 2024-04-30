import numpy as np
from torch.utils.data import Dataset

class eegDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.labels = label
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        
        return data, label
    
    def __len__(self):
        return len(self.data)