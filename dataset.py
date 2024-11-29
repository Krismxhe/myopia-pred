import os
from glob import glob
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import cv2
import pandas as pd
import numpy as np

# define input
# fundus, labels
# define output

class MyopiaFundusDataset(Dataset):
    def __init__(self, file_dir, transform=transforms.ToTensor(), is_train=True):
        """
        label dir: "/data/mengxian/processed_data/002_Myopia_TibetChildren/labels/all_year_no_null_separated"
            stores json file: contains images dir, metadata
        """
        self.file_dir = file_dir
        self.file_name = os.listdir(file_dir)
        self.transform = transform
    
    def __getitem__(self, idx):
        # define json label
        json_file_dir = os.path.join(self.file_dir, self.file_name[idx])
        with open(json_file_dir, 'r') as json_file:
            loaded_data = json.load(json_file)
        # eye's fundus image path
        image_2019 = self.transform(Image.open(loaded_data['fundus']['2019']).convert('RGB'))
        image_2020 = self.transform(Image.open(loaded_data['fundus']['2020']).convert('RGB'))
        image_2021 = self.transform(Image.open(loaded_data['fundus']['2021']).convert('RGB'))
        image_2023 = self.transform(Image.open(loaded_data['fundus']['2023']).convert('RGB'))
        
        # label
        myopia_label = torch.tensor([loaded_data['metadata']['2019']['diopter_param']['myopic'], 
                                     loaded_data['metadata']['2020']['diopter_param']['myopic'],
                                     loaded_data['metadata']['2021']['diopter_param']['myopic'],
                                     loaded_data['metadata']['2023']['diopter_param']['myopic']], dtype=torch.long)
        
        return {'images':torch.stack([image_2019, image_2020, image_2021, image_2023], dim=0), 'label':myopia_label}
        
    def __len__(self):
        return len(self.file_name)

class MyopiaFundusPredDataset(Dataset):
    def __init__(self, 
                 label_dir, 
                 transform=transforms.Compose([transforms.ToTensor(), 
                                               transforms.Resize((256, 256)),
                                               transforms.Normalize((0.5,), (0.5,))]), 
                 is_train=True):

        self.root_data = pd.read_csv(label_dir)
        self.origin_labels = self.root_data['myopic_label'].to_list()
        self.labels = self.origin_labels
        # binary classification experiments
        # for idx, label in enumerate(self.root_data['myopic_label'].to_list()):
        #     self.labels[idx] = 1 if label>2 else 0
        self.imgs_dir = self.root_data['img_dir'].to_list()
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.imgs_dir[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        img = self.transform(Image.open(img_path).convert('RGB'))

        return img, label

    def __len__(self):
        return len(self.labels)

# data augmentation for classification
def aug4cls(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomRotation(degrees=25),
        # transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # keep the gradient, calculate pixel values with corresponding channel, before do the training
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    transform = aug4cls(input_size=(256, 256))
    path = "/data/mengxian/processed_data/002_Myopia_TibetChildren/labels/classification_label/myopia_19_23_cls.csv"
    NUM_WORKERS = os.cpu_count()
    print(f"cpu core number = {NUM_WORKERS}")
    dataset = MyopiaFundusPredDataset(label_dir=path, transform=transform)
    iteration = DataLoader(dataset, batch_size=8, num_workers=NUM_WORKERS)
    for index, batch in enumerate(iteration):
        imgs, labels = batch
        print(imgs.shape)
        print(labels.shape)
