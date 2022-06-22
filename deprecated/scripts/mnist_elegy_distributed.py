import os
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Generator, Mapping, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter

import elegy as eg

import time


class CNN(eg.Module):
    @eg.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Normalize the input
        x = x.astype(jnp.float32) / 255.0

        # Block 1
        x = eg.Conv(32, [3, 3], strides=[2, 2])(x)
        x = eg.Dropout(0.05)(x)
        x = jax.nn.relu(x)

        # Block 2
        x = eg.Conv(64, [3, 3], strides=[2, 2])(x)
        x = eg.BatchNorm()(x)
        x = eg.Dropout(0.1)(x)
        x = jax.nn.relu(x)

        # Block 3
        x = eg.Conv(128, [3, 3], strides=[2, 2])(x)

        # Block 3b
        x = eg.Conv(128, [3, 3], strides=[2, 2])(x)

        # Block 3c
        x = eg.Conv(128, [3, 3], strides=[2, 2])(x)

        # Global average pooling
        x = x.mean(axis=(1, 2))

        # Classification layer
        x = eg.Linear(10)(x)

        return x


def main(
    logdir: str = "runs",
    steps_per_epoch: tp.Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 32
):

    platform = jax.local_devices()[0].platform
    ndevices = len(jax.devices())
    print('devices ', jax.devices())
    print('platform ', platform)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = dataset["train"]["image"][..., None]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["image"][..., None]
    y_test = dataset["test"]["label"]

    accuracies = {}
    # we run distributed=False twice to remove any initial warmup costs
    for distributed in [False,False,True]:
        print(f'Distributed training = {distributed}')
        start_time = time.time()

        model = eg.Model(
            module=CNN(),
            loss=eg.losses.Crossentropy(),
            metrics=eg.metrics.Accuracy(),
            optimizer=optax.adam(1e-3),
            seed = 42
        )

        if distributed:
            model = model.distributed()
            bs = batch_size #int(batch_size / ndevices)
        else:
            bs = batch_size 

        #model.summary(X_train[:64], depth=1)

        history = model.fit(
            inputs=X_train,
            labels=y_train,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=bs,
            validation_data=(X_test, y_test),
            shuffle=True,
            verbose = 3
         )
    
        ev = model.evaluate(x=X_test, y=y_test, verbose=1)
        print('eval ', ev)
        accuracies[distributed] = ev['accuracy']
        
        end_time = time.time()
        print(f'time taken ', {end_time - start_time})

    print(accuracies)


if __name__ == "__main__":
    typer.run(main)
