import torch, torch.nn as nn, torch.nn.functional as F
import os, numpy as np, random
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

def init(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot(embs, targets:np.ndarray):
    import matplotlib.pyplot as plt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        idx = np.where(targets==i)[0]
        plt.scatter(embs[idx, 0], embs[idx, 1], color=colors[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    return fig


class MNIST(Dataset):
    def __init__(self, train:bool=True):
        super().__init__()
        data = datasets.MNIST("../datasets", train)
        self.data = (data.train_data - 0.1307) / 0.3081
        self.data = self.data.unsqueeze(1)
        self.labels = data.train_labels
        self.pos_index = {}
        self.neg_index = {}
        for i in range(10):
            self.pos_index[i] = np.where(self.labels==i)[0]
            self.neg_index[i] = np.where(self.labels!=i)[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx].item()
        x_pos = self.data[random.choice(self.pos_index[label])]
        x_neg = self.data[random.choice(self.neg_index[label])]
        return x, x_pos, x_neg, label


class LitModel(pl.LightningModule):
    def __init__(self, margin, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 256), nn.PReLU(),
            nn.Linear(256, 256), nn.PReLU(),
            nn.Linear(256, 2)
        )
        self.emb = []
        self.labels = []
        self.val_emb = []
        self.val_labels = []

    def forward(self, x):
        return self.net(x)

    def triplet_loss(self, x, x_pos, x_neg):
        dist_pos = (x- x_pos).pow(2).sum(1)
        dist_neg = (x- x_neg).pow(2).sum(1)
        loss = F.relu(dist_pos-dist_neg+self.hparams.margin)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, x_pos, x_neg, label = batch
        out = self(x)
        out_pos = self(x_pos)
        out_neg = self(x_neg)
        self.emb.append(out)
        self.labels.append(label)
        loss = self.triplet_loss(out, out_pos, out_neg)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_pos, x_neg, label = batch
        out = self(x)
        out_pos = self(x_pos)
        out_neg = self(x_neg)
        self.val_emb.append(out)
        self.val_labels.append(label)
        loss = self.triplet_loss(out, out_pos, out_neg)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_epoch_start(self) -> None:
        self.emb = []
        self.labels = []
        self.val_emb = []
        self.val_labels = []

    def on_train_epoch_end(self, outputs) -> None:
        embs = torch.cat(self.emb, 0).cpu().detach().numpy()
        l = torch.cat(self.labels, 0).cpu().detach().numpy()
        fig = plot(embs, l)
        self.logger.experiment.add_figure('figure/train', fig, self.current_epoch)
        self.log('lr', self.optimizers().param_groups[0]['lr'])

    def on_validation_epoch_end(self) -> None:
        embs = torch.cat(self.val_emb, 0).cpu().detach().numpy()
        l = torch.cat(self.val_labels, 0).cpu().detach().numpy()
        fig = plot(embs, l)
        self.logger.experiment.add_figure('figure/val', fig, self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), self.hparams.lr, weight_decay=1e-4)
        schedulers = [StepLR(opt, 50, 0.1)]
        return [opt], schedulers

    def train_dataloader(self) -> DataLoader:
        dataset = MNIST(True)
        return DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        dataset = MNIST(False)
        return DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=1)


if __name__ == "__main__":
    init()
    model = LitModel(1, lr=1e-3)
    logger = TestTubeLogger('lightning_logs', name='triplet', create_git_tag=True)
    trainer = pl.Trainer(logger, gpus=1, max_epochs=100)
    trainer.fit(model)
