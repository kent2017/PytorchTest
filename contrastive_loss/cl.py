# @ref: https://github.com/adambielski/siamese-triplet
import os, numpy as np, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

def init(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot(embeddings, targets):
    import matplotlib.pyplot as plt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        idx = np.where(targets==i)[0]
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], color=colors[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    return fig


class SimpMNIST(Dataset):
    def __init__(self, root, train):
        super().__init__()
        data = datasets.MNIST(root, train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        self.data = (data.data - 0.1307) / 0.3081
        self.data = self.data.unsqueeze(1)
        self.labels = data.targets
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
            nn.Linear(64 * 4 * 4, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2),
        )
        self.emb = []
        self.labels = []
        self.val_emb = []
        self.val_labels = []

    def forward(self, x):
        return self.net(x)

    def contrastive_loss(self, x1, x2, y):
        # y: 0 if sim, 1 if dissim
        dist = torch.sum((x1-x2)**2, dim=1)

        mdist = F.relu(self.hparams.margin-(dist+1e-9)**0.5)        # must add an eps, otherwise the gradient will be nan if dist is 0
        loss = (1-float(y)) * dist + float(y) * (mdist**2)
        loss = (loss*0.5).mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, x_pos, x_neg, label = batch
        out = self(x)
        out_pos = self(x_pos)
        out_neg = self(x_neg)
        loss_s = self.contrastive_loss(out, out_pos, 0)
        loss_d = self.contrastive_loss(out, out_neg, 1)
        self.emb.append(out)
        self.labels.append(label)
        self.log('loss_s', loss_s, True)
        self.log('loss_d', loss_d, True)
        loss = loss_d + loss_s
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_pos, x_neg, label = batch
        out = self(x)
        out_pos = self(x_pos)
        out_neg = self(x_neg)
        loss_s = self.contrastive_loss(out, out_pos, 0)
        loss_d = self.contrastive_loss(out, out_neg, 1)
        self.val_emb.append(out)
        self.val_labels.append(label)
        self.log('val_loss_s', loss_s, True)
        self.log('val_loss_d', loss_d, True)
        loss = loss_d + loss_s
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), self.hparams.lr, weight_decay=1e-4)
        schedulers = [StepLR(opt, 50, 0.1)]
        return [opt], schedulers

    def train_dataloader(self) -> DataLoader:
        dataset = SimpMNIST(root='../datasets', train=True)
        return DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        dataset = SimpMNIST(root='../datasets', train=False)
        return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

    def on_train_epoch_end(self, outputs):
        emb = torch.cat(self.emb, 0)
        l = torch.cat(self.labels, 0)
        self.emb = []
        self.labels = []
        fig = plot(emb.cpu().detach().numpy(), l.cpu().detach().numpy())
        self.logger.experiment.add_figure('figure/train', fig, self.current_epoch)

        self.log('lr', self.optimizers().param_groups[0]['lr'])

    def on_validation_epoch_end(self) -> None:
        emb = torch.cat(self.val_emb, 0)
        l = torch.cat(self.val_labels, 0)
        self.val_emb = []
        self.val_labels = []
        fig = plot(emb.cpu().detach().numpy(), l.cpu().detach().numpy())
        self.logger.experiment.add_figure('figure/val', fig, self.current_epoch)

    def set_hook(self):
        from functools import partial
        # layers = ['0', '2', '4', '6']
        # self.gradients = [None]*len(layers)
        # def fn(i, module, input, output):
        #     self.gradients[i] = input
        #
        # for i in range(len(layers)):
        #     self.net._modules[layers[i]].register_backward_hook(partial(fn, i))
        #

if __name__ == "__main__":
    init()
    model = LitModel(1, lr=1e-3)
    logger = TestTubeLogger('lightning_logs', 'cl', create_git_tag=True)
    # torch.autograd.set_detect_anomaly(True)
    trainer = pl.Trainer(logger, gpus=1, max_epochs=100)
    trainer.fit(model)