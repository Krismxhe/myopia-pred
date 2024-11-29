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

def train_resnet(model,
                 train_dataloader,
                 val_dataloader,
                 max_epochs=1000, 
                 gpus=[0,1,2,3] if torch.cuda.is_available() else None, 
                 **kwargs):

    CHECKPOINT_PATH = '/home/mengxian/002_TibetMyopia/ckpts'
    if os.path.exists(CHECKPOINT_PATH)==False:
        os.mkdir(CHECKPOINT_PATH)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNetPred"),
        accelerator="auto",
        devices=gpus,
        min_epochs=20,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, 
                            mode="max",
                            save_last=True,
                            save_top_k=5,
                            monitor="val_acc"),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_loss", 
                          patience=10,
                          mode="min"),
        ]
    )

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pl.seed_everything(42)  # To be reproducible
    trainer.fit(model, train_loader, val_loader)
    # model = ResNet34Encoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model

if __name__ == "__main__":
    # hyper-param
    NUM_WORKERS = 32
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    MAX_EPOCHS = 1000
    gpus = [0] if torch.cuda.is_available() else None
    # model hyperparam
    LR = 1e-3
    WEIGHT_DECAY = 5e-4
    # define model
    ckpt_path = '/home/mengxian/002_TibetMyopia/ckpts/SimCLR/lightning_logs/version_0/checkpoints/epoch=474-step=4750.ckpt'
    model = ResNet34Encoder(num_classes=NUM_CLASSES, 
                            ckpt_path=ckpt_path,
                            optim='sgd',
                            lr=LR,
                            weight_decay=WEIGHT_DECAY,
                            max_epochs=MAX_EPOCHS,
                            load_params=True)
    # define dataset
    train_data_root_path = "/data/mengxian/processed_data/002_Myopia_TibetChildren/labels/classification_label/trainset.csv"
    val_data_root_path = '/data/mengxian/processed_data/002_Myopia_TibetChildren/labels/classification_label/valset.csv'
    ## define transformation
    input_size = (256, 256)
    transform = aug4cls(input_size=input_size)
    train_dataset = MyopiaFundusPredDataset(label_dir=train_data_root_path, transform=transform)
    val_dataset = MyopiaFundusPredDataset(label_dir=val_data_root_path)
    # avoid distributed sampling, define sampler, dataloader before DDP
    train_loader = GetDataLoader(train_dataset, 
                                 num_workers=NUM_WORKERS, 
                                 batch_size=BATCH_SIZE,
                                 class_weights=[1/7379, 1/1362, 1/388, 1/42],
                                 oversampling=False)
    
    val_loader = GetDataLoader(val_dataset,
                               num_workers=NUM_WORKERS,
                               batch_size=3,
                               class_weights=[1/7379, 1/1362, 1/388, 1/42],
                               oversampling=False)
    # DDP training
    resnet_model = train_resnet(model=model,
                                train_dataloader=train_loader,
                                val_dataloader=val_loader,
                                max_epochs=MAX_EPOCHS, 
                                gpus=gpus,
                                lr=1e-3, 
                                weight_decay=2e-4)