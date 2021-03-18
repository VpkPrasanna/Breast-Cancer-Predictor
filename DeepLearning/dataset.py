import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import cv2

TrainPath = ""
class BreastTumor(Dataset):
    def __init__(self,df,transforms=None):
        self.df = df
        self.file_names = df["imageid"]
        self.targets = df["target"]
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        file_name = self.file_names[idx]
        file_path = f'{TrainPath}/{file_name}'
        image = cv2.imread(file_path,cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        label = torch.tensor(self.targets[idx]).long()
        return image,label