import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl

import os
from tqdm import tqdm
from dataset import MyopiaFundusPredDataset, aug4cls

def GetDataLoader(dataset, 
                  num_workers, 
                  batch_size,
                  class_weights = [1/7379, 1/1362, 1/388, 1/42],
                  oversampling=True):
                  
    if oversampling==True:
        class_weights = class_weights
        sample_weights = [0] * len(dataset)
        for idx, label in tqdm(enumerate(dataset.labels)):
            # specify the sample weight according to hyper-params
            class_weight = class_weights[int(label)]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, 
                                        num_samples=len(sample_weights),
                                        replacement=True)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers
        ) 

        return dataloader

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers
        )

        return dataloader

if __name__ == "__main__":
    # hyper-param
    NUM_WORKERS = int(os.cpu_count() / 2)
    BATCH_SIZE = 256
    # define dataset
    train_data_root_path = "/data/mengxian/processed_data/002_Myopia_TibetChildren/labels/classification_label/trainset.csv"
    ## define transformation
    input_size = (256, 256)
    transform = aug4cls(input_size=input_size)
    train_dataset = MyopiaFundusPredDataset(label_dir=train_data_root_path, transform=transform)
    # avoid distributed sampling, define sampler, dataloader before DDP
    train_loader = GetDataLoader(train_dataset, 
                                 num_workers=NUM_WORKERS, 
                                 batch_size=BATCH_SIZE)
    
    for index, batch in enumerate(train_loader):
        imgs, labels = batch
        print(imgs.shape)
        print(labels.shape)