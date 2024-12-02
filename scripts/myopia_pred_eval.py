import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
from tqdm import tqdm
from dataset import MyopiaFundusPredDataset, aug4cls
from dataloader import GetDataLoader
from models import ResNet34Encoder

def eval_resnet(model,
                test_dataloader,
                gpus=[0,1,2,3] if torch.cuda.is_available() else None, 
                **kwargs):
    EVAL_PATH = '/example'
    trainer = pl.Trainer(default_root_dir=os.path.join(EVAL_PATH, "ResNetPred"),
                         devices=gpus
    )
    test_result = trainer.test(model=model, dataloaders=test_dataloader)
    return test_result

if __name__ == "__main__":
    gpus = [1] if torch.cuda.is_available() else None
    # define model
    ckpt_path = '/example'
    model = ResNet34Encoder.load_from_checkpoint(ckpt_path)
    # define dataset
    test_data_root_path = '/example'
    test_dataset = MyopiaFundusPredDataset(label_dir=test_data_root_path)
    # avoid distributed sampling, define sampler, dataloader before DDP
    test_loader = GetDataLoader(test_dataset, 
                                num_workers=8, 
                                batch_size=8,
                                class_weights=[1/7379, 1/1792],
                                oversampling=True)
                                 
    # eval
    test_result = eval_resnet(model=model,
                              test_dataloader=test_loader,
                              gpus=gpus)
    print(test_result)