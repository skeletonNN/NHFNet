import os
import torch
import numpy as np
from torch.utils.data import Dataset

"""
CMU-MOSEI info
Train 16322 samples
Val 1871 samples
Test 4659 samples

CMU-MOSEI feature shapes
visual: (50, 35)
audio: (50, 74)
text: GLOVE->(50, 300), BERT->(50, 768)
label: (1, 7) -> [sentiment, happy, sad, anger, surprise, disgust, fear] 
    sentiment in range [-3, 3] 
    emotions in range [0, 3]
    averaged from 3 annotators
"""


class CMUMOSEIDataset(Dataset):
    """CMU-MOSEI dataset."""

    def __init__(self, dataset_folder, param):
        """
        :param dataset_folder: train/val/test folder path
        :param param: dict with filenames (without .npy) of each modality and label. Two options for labels
            e.g. param = {
                "audio": None,
                "visual": None,
                "text" or "bert": None,
                "label": {
                    "sentiment" or "emotion"
                }
            }
        """
        self.data = []
        self.param = param

        for p in param:
            # if p == "label" and "sentimentx" in param[p]:
            if p == "label" and "sentiment" in param[p]:
                self.data.append(np.load(os.path.join(dataset_folder, p + "50.npy"))[:, :, 0])
            elif p == "label" and "emotion" in param[p]:
                self.data.append(np.load(os.path.join(dataset_folder, p + "50.npy"))[:, :, 1:])
            else:
                self.data.append(np.load(os.path.join(dataset_folder, p + "50.npy")))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = {}

        for data, p in zip(self.data, self.param):
            value = torch.tensor(data[idx])
            value[torch.isinf(value)] = 0
            sample.update({p: value})
        return sample
