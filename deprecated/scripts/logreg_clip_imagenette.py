"""
This file illustrates logistic regression for CLIP extracted features of imagenette dataset
using sklearn and pytorch_lightning
Author : Srikar-Reddy-Jilugu(@always-newbie161)
"""

import superimport

from clip_dataloader import get_imagenette_clip_loaders
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

train_loader, test_loader = get_imagenette_clip_loaders(train_shuffle=True)


def convert_dataloader_to_numpy(loader):
    features, labels = [], []

    for batch in loader:
        features.append(batch[0])
        labels.append(batch[1])

    features, labels = np.concatenate(features), np.concatenate(labels)

    return features, labels


# -----------
# logReg using sklearn
def run_sklearn(train_loader, test_loader):
    train_features, train_labels = convert_dataloader_to_numpy(train_loader)
    test_features, test_labels = convert_dataloader_to_numpy(test_loader)
    clf = LogisticRegression(random_state=0, C=1, solver='saga', max_iter=500)
    print('Training sklearn model...')
    clf.fit(train_features, train_labels)
    train_preds_skl = clf.predict(train_features)
    test_preds_skl = clf.predict(test_features)
    test_accuracy = np.mean((test_labels == test_preds_skl).astype(np.float64)) * 100.
    train_accuracy = np.mean((train_labels == train_preds_skl).astype(np.float64)) * 100.
    print(f"skl_train_accuracy: {train_accuracy:.3f}, skl_test_accuracy: {test_accuracy:.3f}")
    return train_preds_skl, test_preds_skl


# ------------
# logReg using pytorch_lightning


class Lit_model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        out = self.network(x)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        out = self.network(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        self.log('val_loss', loss)


def lit_predict(model, x):
    x = x.view(x.size(0), -1)
    out = model(x)
    logits = nn.LogSoftmax(dim=-1)(out)
    ypred = torch.argmax(logits, dim=-1)
    return ypred.numpy()


def run_lightning(args, train_loader, test_loader):
    print('Training pytorch_lightning model...')
    model = Lit_model()

    # training (can be interrupted safely by ctrl+C or "Interrupt execution" on colab)
    trainer = pl.Trainer(max_epochs=5, deterministic=True, auto_select_gpus=True)
    trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, test_loader)

    train_features, train_labels = convert_dataloader_to_numpy(train_loader)
    test_features, test_labels = convert_dataloader_to_numpy(test_loader)

    model.freeze()
    train_preds_lit = lit_predict(model, torch.Tensor(train_features))
    test_preds_lit = lit_predict(model, torch.Tensor(test_features))

    train_accuracy = np.mean((train_labels == train_preds_lit)) * 100.
    test_accuracy = np.mean((test_labels == test_preds_lit)) * 100.
    print(f"lit_train_accuracy: {train_accuracy:.3f}, lit_test_accuracy: {test_accuracy:.3f}")

    return train_preds_lit, test_preds_lit


def main(args):
    pl.seed_everything(42, workers=True)
    train_preds_skl, test_preds_skl = run_sklearn(train_loader, test_loader)
    train_preds_lit, test_preds_lit = run_lightning(args, train_loader, test_loader)
    pred_diff = np.mean((test_preds_lit != test_preds_skl))
    assert np.allclose(pred_diff, np.zeros(pred_diff.shape), atol=1e-2)


# arguments to the Trainer can also be passed when about to run the script
# for Eg: $ python3 imagenette_clip_logreg.py --gpus 2 --max_steps 10 --limit_train_batches 10
# So you can mention no.of gpus or tpu cores of the machine you are running on as the flags.
# (See pl.Trainer doc for all possible arguments)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
