# from 
# https://examples.dask.org/machine-learning/torch-prediction.html

####################
# Setup coiled/ dask

import coiled

cluster = coiled.Cluster(
    n_workers=2, #10
    software="examples/hyperband-optimization",
)


import dask.distributed
client = dask.distributed.Client(cluster)


##from distributed import Client
#client = Client(n_workers=2, threads_per_worker=2)


print(client)


####################
# Download data

import urllib.request
import zipfile

filename, _ = urllib.request.urlretrieve("https://download.pytorch.org/tutorial/hymenoptera_data.zip", "data.zip")
zipfile.ZipFile(filename).extractall()
  


############
# Pytorch model

import torchvision
from tutorial_helper import (imshow, train_model, visualize_model,
                             dataloaders, class_names, finetune_model)

import dask

model = finetune_model()

visualize_model(model)


#########
# Remote prediction


###########
# Loading the data on the workers


import glob
import toolz
import dask
import dask.array as da
import torch
from torchvision import transforms
from PIL import Image


@dask.delayed
def load(path, fs=__builtins__):
    with fs.open(path, 'rb') as f:
        img = Image.open(f).convert("RGB")
        return img


@dask.delayed
def transform(img):
    trn = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return trn(img)

objs = [load(x) for x in glob.glob("hymenoptera_data/val/*/*.jpg")]

tensors = [transform(x) for x in objs]

batches = [dask.delayed(torch.stack)(batch)
           for batch in toolz.partition_all(10, tensors)]

print(batches[:5])


@dask.delayed
def predict(batch, model):
    with torch.no_grad():
        out = model(batch)
        _, predicted = torch.max(out, 1)
        predicted = predicted.numpy()
    return predicted


############
# Moving the model around



import pickle

print(dask.utils.format_bytes(len(pickle.dumps(model))))



dmodel = dask.delayed(model.cpu()) # ensuring model is on the CPU

predictions = [predict(batch, dmodel) for batch in batches]
#dask.visualize(predictions[:2])

# fails
# [Errno 2] No such file or directory: 'hymenoptera_data/val/ants/181942028_961261ef48.jpg'
predictions = dask.compute(*predictions)
predictions

