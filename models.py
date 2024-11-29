import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
from tqdm.notebook import tqdm
from focal_loss.focal_loss import FocalLoss
from simclr import SimCLR

class ResNet34Encoder(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 ckpt_path=None,
                 optim='adawW',
                 hidden_dim=128, 
                 lr=1e-3, 
                 weight_decay=2e-4,
                 max_epochs=1000,
                 load_params=True):
        super().__init__()
        self.save_hyperparameters()
        self.optim=optim
        self.convnet = None
        if load_params==True:
            backbone = SimCLR.load_from_checkpoint(ckpt_path).convnet
            self.convnet = backbone
            self.convnet.fc = nn.Sequential(
                nn.Linear(4 * 128, 4 * 128),
                nn.ReLU(inplace=True),
                nn.Linear(4 * 128, num_classes),
            )
        else:
            backbone = torchvision.models.resnet34(pretrained=False, 
                                                   num_classes=4 * 128)
            backbone.fc = nn.Sequential(
                backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4 * 128, 128),
            )
            self.convnet = backbone
            self.convnet.fc[2] = nn.Linear(4 * 128, num_classes)
        
        # evaluation
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.prec = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.rec = Recall(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        preds = self.convnet(x)
        return preds
    
    def ce_loss(self, preds, labels):
        return F.cross_entropy(preds, labels)
    
    def focal_loss(self, preds, labels, gamma=0.7):
        criterion = FocalLoss(gamma=gamma)
        focal_loss = criterion(preds, labels)
        return focal_loss

    def _calculate_loss(self, batch, mode='train'):
        # input
        #   imgs: torch.Size([8, 4, 3, 256, 256])
        #   label: torch.Size([8, 4])
        imgs, labels = batch
        preds = self.convnet(imgs)
        y_preds = torch.softmax(preds, dim=1)
        loss = self.ce_loss(y_preds, labels)
        acc = self.acc(y_preds, labels)
        f1 = self.f1(y_preds, labels)

        if mode=='train':
            self.log(mode + "_loss", loss, sync_dist=True)
            self.log(mode + "_acc", acc, sync_dist=True)
            self.log(mode + "_f1", f1, sync_dist=True)
        elif mode=='val':
            auroc = self.auroc(y_preds, labels)
            # prec = self.prec(y_preds, labels)
            # rec = self.rec(y_preds, labels)

            self.log(mode + "_loss", loss, sync_dist=True)
            self.log(mode + "_acc", acc, sync_dist=True)
            self.log(mode + "_f1", f1, sync_dist=True)
            self.log(mode + "_auroc", auroc, sync_dist=True)
            # self.log(mode + "_precision", prec, sync_dist=True)
            # self.log(mode + "_recall", rec, sync_dist=True)
        elif mode=='test':
            auroc = self.auroc(y_preds, labels)
            prec = self.prec(y_preds, labels)
            rec = self.rec(y_preds, labels)

            self.log(mode + "_loss", loss, sync_dist=True)
            self.log(mode + "_acc", acc, sync_dist=True)
            self.log(mode + "_f1", f1, sync_dist=True)
            self.log(mode + "_auroc", auroc, sync_dist=True)
            self.log(mode + "_precision", prec, sync_dist=True)
            self.log(mode + "_recall", rec, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = None
        if self.optim=='sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[int(5), int(20)],
            gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')
    
    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')

class ResNet34(pl.LightningModule):

    def __init__(self, num_classes, lr=1e-3, weight_decay=2e-4, max_epochs=1000):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet34(pretrained=False, num_classes=4)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.hparams.max_epochs * 0.7), int(self.hparams.max_epochs * 0.9)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class ImageSequenceModel(nn.Module):
    def __init__(self, cnn_encoder, seq_len, hidden_size, output_size):
        super(ImageSequenceModel, self).__init__()
        self.cnn_encoder = cnn_encoder
        self.seq_len = seq_len
        self.rnn = nn.LSTM(input_size=512+1, hidden_size=hidden_size, batch_first=True)  # [batch_size, seq_len, 513] -> [batch_size, seq_len, hidden_size]
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, output_size)  # [batch_size, hidden_size] -> [batch_size, output_size]
        

    def forward(self, image_sequence, feature_sequence):
        batch_size, _, _, _, _ = image_sequence.size()  # [batch_size, seq_len, 3, 512, 512]
        cnn_features = []
        for t in range(self.seq_len):
            x = image_sequence[:, t, :, :, :]  # [batch_size, 3, 512, 512]
            cnn_out = self.cnn_encoder(x)  # [batch_size, 512]
            cnn_features.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, seq_len, 512]
        rnn_input = torch.cat([cnn_features, feature_sequence], dim=-1)  # [batch_size, seq_len, 513]
        rnn_out, (h_n, c_n) = self.rnn(rnn_input)  # rnn_out: [batch_size, seq_len, hidden_size], h_n: [batch_size, 1, hidden_size]
        #rnn_out, (h_n, c_n) = self.rnn(cnn_features)
        h_n = self.dropout(h_n)
        output = self.linear(h_n.squeeze(0))  # [batch_size, hidden_size] -> [batch_size, output_size]
        return output

if __name__  == "__main__":
    print('hello world!')