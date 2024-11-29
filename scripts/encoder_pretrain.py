import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from tqdm.notebook import tqdm
from simclr import SimCLR, ContrastiveTransformations
import os

def train_simclr(batch_size=256, max_epochs=500, gpus=[0, 1, 2, 3] if torch.cuda.is_available() else None, **kwargs):
    CHECKPOINT_PATH = '/home/mengxian/002_TibetMyopia/ckpts'
    if os.path.exists(CHECKPOINT_PATH)==False:
        os.mkdir(CHECKPOINT_PATH)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="auto",
        devices=gpus,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(os.path.join(CHECKPOINT_PATH, 'SimCLR'), "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        NUM_WORKERS = os.cpu_count()
        path = '/data/mengxian/processed_data/002_Myopia_TibetChildren/images/'
        dataset = ImageFolder(root=path, transform=ContrastiveTransformations())
        dataset.imgs = [(path, 0) for path, _ in dataset.imgs]
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        pl.seed_everything(42)  # To be reproducible
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model

if __name__ == "__main__":
    gpus = [0, 1, 2, 3] if torch.cuda.is_available() else None
    simclr_model = train_simclr(
        batch_size=256, 
        max_epochs=500, 
        gpus=gpus, 
        hidden_dim=128, 
        lr=5e-4, 
        temperature=0.07, 
        weight_decay=1e-4
    )