# Based on 
# https://github.com/williamFalcon/cifar5-simple/blob/main/cifar5.py
'''
python3 pyprobml/scripts/cifar5-grid.py \
  --data_dir cifar5 --gpus=1 --log_every_n_steps=1  \
  --max_epochs=3 --limit_train_batches=2 --limit_val_batches=2 --limit_test_batches=2
'''

from argparse import ArgumentParser

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as PLF
from torch.nn import functional as F
from flash.vision import ImageClassificationData
from torchvision import transforms
from torchvision import models
import numpy as np


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone='resnet50', num_classes=5, hidden_dim=1024, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = getattr(models, backbone)()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, hidden_dim),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch):
        x, y = batch

        # used only in .predict()
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        predicted_classes = F.log_softmax(y_hat, dim=1).argmax(dim=1)
        return predicted_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        acc = PLF.accuracy(F.log_softmax(y_hat, dim=1).argmax(dim=1), y)
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        acc = PLF.accuracy(F.log_softmax(y_hat, dim=1).argmax(dim=1), y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--backbone', type=str, default='resnet50')
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_classes', default=5, type=int)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='cifar5')
    #parser.add_argument('--max_epochs', type=int, default=2)

    # add trainer args (gpus=x, precision=...)
    parser = pl.Trainer.add_argparse_args(parser)

    # add model args (batch_size hidden_dim, etc...), anything defined in add_model_specific_args
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    # ------------
    # data
    # ------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4913, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # in real life you would have a separate validation split
    datamodule = ImageClassificationData.from_folders(
        train_folder=args.data_dir + '/train',
        valid_folder=args.data_dir + '/test',
        test_folder=args.data_dir + '/test',
        batch_size=args.batch_size,
        transform=transform
    )

    # ------------
    # model
    # ------------
    model = LitClassifier(
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim
    )

    # ------------
    # training
    # ------------
    print('training')
    trainer = pl.Trainer.from_argparse_args(args) #, fast_dev_run=True)
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    
    # ------------
    # testing
    # ------------
    print('testing')
    result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    print(result)

    # predicting
    print('predicting')
    preds = trainer.predict(model, datamodule.test_dataloader())
    #import pdb; pdb.set_trace()
    #print(preds) # list of n=N/B tensors, each of size B=batchsize=32.
    #preds = list(np.stack(preds).flatten()) # fails on last batch, which is shorter

    path = os.getcwd() + '/predictions.txt'
    with open(path, 'w') as f:
        preds_str = [str(x) for lst in preds for x in lst]
        f.write('\n'.join(preds_str))


if __name__ == '__main__':
    cli_main()