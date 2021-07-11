# based on https://github.com/williamFalcon/hello/blob/main/hello.py
import os
import numpy as np
from argparse import ArgumentParser
import torch
from torch.utils.tensorboard import SummaryWriter

# add arguments 
parser = ArgumentParser()
parser.add_argument('--number', default=0, type=int)
parser.add_argument('--food_item', default='burgers', type=str)
parser.add_argument('--data', default=None, type=str)
args = parser.parse_args()
print(args)

# fake tensorboard logs (fake loss)
writer = SummaryWriter()
offset = np.random.uniform(0, 5, 1)[0]
for x in range(1, 10000):
    y = -np.log(x) + offset + (np.sin(x) * 0.1)
    writer.add_scalar('y=-log(x) + c + 0.1sin(x)', y, x)
    writer.add_scalar('fake_metric', -y, x)

writer.close()

# print data if available
if args.data is not None:
    files = list(os.walk(args.data))
    print('-' * 50)
    print(f'DATA FOUND! {len(files)} files found at dataset {args.data}')

# print GPUs, params and random tensors
print('-' * 50)
print(f'GPUS: There are {torch.cuda.device_count()} GPUs on this machine')
print('-' * 50)
print(f'PARAMS: I want to eat: {args.number} {args.food_item}')
print('-' * 50)
print('i can run any ML library like numpy, pytorch lightning, sklearn pytorch, keras, tensorflow')
print('torch:', torch.rand(1), 'numpy', np.random.rand(1))

# write some artifacts
f = open("weights.pt", "a")
f.write("fake weights")
f.close()
