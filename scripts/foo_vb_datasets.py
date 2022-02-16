'''
Reference
     https://github.com/chenzeno/FOO-VB/blob/ebc14a930ba9d1c1dadc8e835f746c567c253946/datasets.py
'''

import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
import os
import codecs
from torch.distributions.categorical import Categorical
import torch.utils.data as data
from PIL import Image
import errno


def _reduce_class(set, classes, train, preserve_label_space=True):
    if classes is None:
        return

    new_class_idx = {}
    for c in classes:
        new_class_idx[c] = new_class_idx.__len__()

    new_data = []
    new_labels = []
    if train:
        all_data = set.train_data
        labels = set.train_labels
    else:
        all_data = set.test_data
        labels = set.test_labels

    for data, label in zip(all_data, labels):
        if type(label) == int:
            label_val = label
        else:
            label_val = label.item()
        if label_val in classes:
            new_data.append(data)
            if preserve_label_space:
                new_labels += [label_val]
            else:
                new_labels += [new_class_idx[label_val]]
    if type(new_data[0]) == np.ndarray:
        new_data = np.array(new_data)
    elif type(new_data[0]) == torch.Tensor:
        new_data = torch.stack(new_data)
    else:
        assert False, "Reduce class not supported"
    if train:
        set.train_data = new_data
        set.train_labels = new_labels
    else:
        set.test_data = new_data
        set.test_labels = new_labels


class Permutation(torch.utils.data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, permute_idx, target_offset):
        super(Permutation,self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx
        self.target_offset = target_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target = target + self.target_offset
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target


class DatasetsLoaders:
    def __init__(self, dataset, batch_size=4, num_workers=4, pin_memory=True, **kwargs):
        self.dataset_name = dataset
        self.valid_loader = None
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 4

        self.random_erasing = kwargs.get("random_erasing", False)
        self.reduce_classes = kwargs.get("reduce_classes", None)
        self.permute = kwargs.get("permute", False)
        self.target_offset = kwargs.get("target_offset", 0)

        pin_memory = pin_memory if torch.cuda.is_available() else False
        self.batch_size = batch_size
        cifar10_mean = (0.5, 0.5, 0.5)
        cifar10_std = (0.5, 0.5, 0.5)
        cifar100_mean = (0.5070, 0.4865, 0.4409)
        cifar100_std = (0.2673, 0.2564, 0.2761)
        mnist_mean = [33.318421449829934]
        mnist_std = [78.56749083061408]
        fashionmnist_mean = [73.14654541015625]
        fashionmnist_std = [89.8732681274414]

        if dataset == "CIFAR10":
            # CIFAR10:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 125.306918046875
            #                        train_set.train_data[:,:,:,1].mean() 122.95039414062499
            #                        train_set.train_data[:,:,:,2].mean() 113.86538318359375
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 62.993219278136884
            #                        train_set.train_data[:, :, :, 1].std() 62.088707640014213
            #                        train_set.train_data[:, :, :, 2].std() 66.704899640630913
            self.mean = cifar10_mean
            self.std = cifar10_std

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                          download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                         download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "CIFAR100":
            # CIFAR100:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 129.304165605/255=0.5070
            #                        train_set.train_data[:,:,:,1].mean() 124.069962695/255=0.4865
            #                        train_set.train_data[:,:,:,2].mean() 112.434050059/255=0.4409
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 68.1702428992/255=0.2673
            #                        train_set.train_data[:, :, :, 1].std() 65.3918080439/255=0.2564
            #                        train_set.train_data[:, :, :, 2].std() 70.418370188/255=0.2761

            self.mean = cifar100_mean
            self.std = cifar100_std
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(self.mean, self.std)])

            self.train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                           download=True, transform=transform)
            _reduce_class(self.train_set, self.reduce_classes, train=True,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                          download=True, transform=transform)
            _reduce_class(self.test_set, self.reduce_classes, train=False,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "MNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : 33.318421449829934
            #   Std per channel    : 78.56749083061408

            # Transforms
            self.mean = mnist_mean
            self.std = mnist_std
            if kwargs.get("pad_to_32", False):
                transform = transforms.Compose(
                    [transforms.Pad(2, fill=0, padding_mode='constant'),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.1000,), std=(0.2752,))])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor()])

            # Create train set
            self.train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                                        download=True, transform=transform)
            if kwargs.get("permutation", False):
                # Permute if permutation is provided
                self.train_set = Permutation(torchvision.datasets.MNIST(root='./data', train=True,
                                                                        download=True, transform=transform),
                                             kwargs.get("permutation", False), self.target_offset)
            # Reduce classes if necessary
            _reduce_class(self.train_set, self.reduce_classes, train=True,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            # Remap labels
            if kwargs.get("labels_remapping", False):
                labels_remapping = kwargs.get("labels_remapping", False)
                for lbl_idx in range(len(self.train_set.train_labels)):
                    self.train_set.train_labels[lbl_idx] = labels_remapping[self.train_set.train_labels[lbl_idx]]

            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            # Create test set
            self.test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                                       download=True, transform=transform)
            if kwargs.get("permutation", False):
                # Permute if permutation is provided
                self.test_set = Permutation(torchvision.datasets.MNIST(root='./data', train=False,
                                                                        download=True, transform=transform),
                                             kwargs.get("permutation", False), self.target_offset)
            # Reduce classes if necessary
            _reduce_class(self.test_set, self.reduce_classes, train=False,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            # Remap labels
            if kwargs.get("labels_remapping", False):
                labels_remapping = kwargs.get("labels_remapping", False)
                for lbl_idx in range(len(self.test_set.test_labels)):
                    self.test_set.test_labels[lbl_idx] = labels_remapping[self.test_set.test_labels[lbl_idx]]

            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "FashionMNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : fm.train_data.type(torch.FloatTensor).mean() is 72.94035223214286
            #   Std per channel    : fm.train_data.type(torch.FloatTensor).std() is 90.0211833054075
            self.mean = fashionmnist_mean
            self.std = fashionmnist_std
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize(self.mean, self.std)])
            # transform = transforms.Compose(
            #     [transforms.ToTensor()])
            transform = transforms.Compose(
                [transforms.Pad(2),
                 transforms.ToTensor(),
                 transforms.Normalize((72.94035223214286 / 255,), (90.0211833054075 / 255,))])



            self.train_set = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=True,
                                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=False,
                                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "SVHN":
            # SVHN:
            #   type               : numpy.ndarray
            #   shape              : self.train_set.data.shape is (73257, 3, 32, 32)
            #   test data shape    : self.test_set.data.shape is (26032, 3, 32, 32)
            #   number of channels : 3
            #   Mean per channel   : sv.data.mean(axis=0).mean(axis=1).mean(axis=1) is array([111.60893668, 113.16127466, 120.56512767])
            #   Std per channel    : np.transpose(sv.data, (1, 0, 2, 3)).reshape(3,-1).std(axis=1) is array([50.49768174, 51.2589843 , 50.24421614])
            self.mean = mnist_mean
            self.std = mnist_std
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize(self.mean, self.std)])
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((111.60893668/255, 113.16127466/255, 120.56512767/255), (50.49768174/255, 51.2589843/255, 50.24421614/255))])



            self.train_set = torchvision.datasets.SVHN(root='./data', split="train",
                                                               download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.SVHN(root='./data', split="test",
                                                              download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "NOTMNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : nm.train_data.type(torch.FloatTensor).mean() is 106.51712372448979
            #   Std per channel    : nm.train_data.type(torch.FloatTensor).std() is 115.76734631096612
            self.mean = mnist_mean
            self.std = mnist_std
            transform = transforms.Compose(
                [transforms.Pad(2),
                 transforms.ToTensor(),
                 transforms.Normalize((106.51712372448979 / 255,), (115.76734631096612 / 255,))])

            self.train_set = NOTMNIST(root='./data/notmnist', train=True, download=True, transform=transform)

            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=pin_memory)

            self.test_set = NOTMNIST(root='./data/notmnist', train=False, download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1,
                                                           pin_memory=pin_memory)
        if dataset == "CONTPERMUTEDPADDEDMNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])
            '''
            transform = transforms.Compose(
                [transforms.Pad(2, fill=0, padding_mode='constant'),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.1000,), std=(0.2752,))])
            '''

            # Original MNIST
            tasks_datasets = [torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)]
            tasks_samples_indices = [torch.tensor(range(len(tasks_datasets[0])), dtype=torch.int32)]
            total_len = len(tasks_datasets[0])
            test_loaders = [torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False,
                                                                                   download=True, transform=transform),
                                                        batch_size=self.batch_size, shuffle=False,
                                                        num_workers=1, pin_memory=pin_memory)]
            self.num_of_permutations = len(kwargs.get("all_permutation"))
            all_permutation = kwargs.get("all_permutation", None)
            for p_idx in range(self.num_of_permutations):
                # Create permuation
                permutation = all_permutation[p_idx]

                # Add train set:
                tasks_datasets.append(Permutation(torchvision.datasets.MNIST(root='./data', train=True,
                                                                             download=True, transform=transform),
                                                  permutation, target_offset=0))

                tasks_samples_indices.append(torch.tensor(range(total_len,
                                                                total_len + len(tasks_datasets[-1])
                                                                ), dtype=torch.int32))
                total_len += len(tasks_datasets[-1])
                # Add test set:
                test_set = Permutation(torchvision.datasets.MNIST(root='./data', train=False,
                                                                  download=True, transform=transform),
                                       permutation, self.target_offset)
                test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                shuffle=False, num_workers=1,
                                                                pin_memory=pin_memory))
            self.test_loader = test_loaders
            # Concat datasets
            total_iters = kwargs.get("total_iters", None)

            assert total_iters is not None
            beta = kwargs.get("contpermuted_beta", 3)
            all_datasets = torch.utils.data.ConcatDataset(tasks_datasets)

            # Create probabilities of tasks over iterations
            self.tasks_probs_over_iterations = [_create_task_probs(total_iters, self.num_of_permutations+1, task_id,
                                                                    beta=beta) for task_id in
                                                 range(self.num_of_permutations+1)]
            normalize_probs = torch.zeros_like(self.tasks_probs_over_iterations[0])
            for probs in self.tasks_probs_over_iterations:
                normalize_probs.add_(probs)
            for probs in self.tasks_probs_over_iterations:
                probs.div_(normalize_probs)
            self.tasks_probs_over_iterations = torch.cat(self.tasks_probs_over_iterations).view(-1, self.tasks_probs_over_iterations[0].shape[0])
            tasks_probs_over_iterations_lst = []
            for col in range(self.tasks_probs_over_iterations.shape[1]):
                tasks_probs_over_iterations_lst.append(self.tasks_probs_over_iterations[:, col])
            self.tasks_probs_over_iterations = tasks_probs_over_iterations_lst

            train_sampler = ContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                         tasks_samples_indices=tasks_samples_indices,
                                                         tasks_probs_over_iterations=
                                                             self.tasks_probs_over_iterations,
                                                         num_of_batches=kwargs.get("iterations_per_virtual_epc", 1))
            self.train_loader = torch.utils.data.DataLoader(all_datasets, batch_size=self.batch_size,
                                                            num_workers=1, sampler=train_sampler, pin_memory=pin_memory)


class ContinuousMultinomialSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    self.tasks_probs_over_iterations is the probabilities of tasks over iterations.
    self.samples_distribution_over_time is the actual distribution of samples over iterations
                                            (the result of sampling from self.tasks_probs_over_iterations).
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, samples_in_batch=128, num_of_batches=69, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):
        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.num_of_batches = num_of_batches

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))

    def generate_iters_indices(self, num_of_iters):
        from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, from_iter+num_of_iters):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return len(self.samples_in_batch)


def _get_linear_line(start, end, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i - start)/(end-start) for i in range(start, end)])
    return torch.FloatTensor([1 - ((i - start) / (end - start)) for i in range(start, end)])


def _create_task_probs(iters, tasks, task_id, beta=3):
    if beta <= 1:
        peak_start = int((task_id/tasks)*iters)
        peak_end = int(((task_id + 1) / tasks)*iters)
        start = peak_start
        end = peak_end
    else:
        start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
        peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
        peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
        end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

    probs = torch.zeros(iters, dtype=torch.float)
    if task_id == 0:
        probs[start:peak_start].add_(1)
    else:
        probs[start:peak_start] = _get_linear_line(start, peak_start, direction="up")
    probs[peak_start:peak_end].add_(1)
    if task_id == tasks - 1:
        probs[peak_end:end].add_(1)
    else:
        probs[peak_end:end] = _get_linear_line(peak_end, end, direction="down")
    return probs


###
# NotMNIST
###
class NOTMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-images-idx3-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-labels-idx1-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-images-idx3-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            self.read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            self.read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            self.read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            self.read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    def read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2049
            length = self.get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    def read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2051
            length = self.get_int(data[4:8])
            num_rows = self.get_int(data[8:12])
            num_cols = self.get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)


###########################################################################
# Callable datasets
###########################################################################


def ds_mnist(**kwargs):
    """
    MNIST dataset.
    :param batch_size: batch size
           num_workers: num of workers
           pad_to_32: If true, will pad digits to size 32x32 and normalize to zero mean and unit variance.
    :return: Tuple with two lists.
             First list of the tuple is a list of 1 train loaders.
             Second list of the tuple is a list of 1 test loaders.
    """
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               pad_to_32=kwargs.get("pad_to_32", False))]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_split_mnist(**kwargs):
    """
    Split MNIST dataset. Consists of 5 tasks: digits 0 & 1, 2 & 3, 4 & 5, 6 & 7, and 8 & 9.
    :param batch_size: batch size
           num_workers: num of workers
           pad_to_32: If true, will pad digits to size 32x32 and normalize to zero mean and unit variance.
           separate_labels_space: If true, each task will have its own label space (e.g. 01, 23 etc.).
                                  If false, all tasks will have label space of 0,1 only.
    :return: Tuple with two lists.
             First list of the tuple is a list of 5 train loaders, each loader is a task.
             Second list of the tuple is a list of 5 test loaders, each loader is a task.
    """
    classes_lst = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ]
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               reduce_classes=cl, pad_to_32=kwargs.get("pad_to_32", False),
                               preserve_label_space=kwargs.get("separate_labels_space")) for cl in classes_lst]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_padded_split_mnist(**kwargs):
    """
    Split MNIST dataset, padded to 32x32 pixels.
    """
    return ds_split_mnist(pad_to_32=True, **kwargs)


def ds_split_mnist_offline(**kwargs):
    """
    Split MNIST dataset. Offline means that all tasks are mixed together.
    """
    if kwargs.get("separate_labels_space"):
        return ds_mnist(**kwargs)
    else:
        return ds_mnist(labels_remapping={l: l % 2 for l in range(10)}, **kwargs)


def ds_padded_split_mnist_offline(**kwargs):
    """
    Split MNIST dataset. Padded to 32x32. Offline means that all tasks are mixed together.
    """
    return ds_split_mnist_offline(pad_to_32=True, **kwargs)


def ds_permuted_mnist(**kwargs):
    """
    Permuted MNIST dataset.
    First task is the MNIST datasets (with 10 possible labels).
    Other tasks are permutations (pixel-wise) of the MNIST datasets (with 10 possible labels).
    :param batch_size: batch size
           num_workers: num of workers
           pad_to_32: If true, will pad digits to size 32x32 and normalize to zero mean and unit variance.
           permutations: A list of permutations. Each permutation should be a list containing new pixel position.
           separate_labels_space: True for seperated labels space - task i labels will be (10*i) to (10*i + 9).
                                  False for unified labels space - all tasks will have labels of 0 to 9.
    :return: Tuple with two lists.
             First list of the tuple is a list of train loaders, each loader is a task.
             Second list of the tuple is a list of test loaders, each loader is a task.
    """
    # First task
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), pad_to_32=kwargs.get("pad_to_32", False))]
    target_offset = 0
    permutations = kwargs.get("permutations", [])
    for pidx in range(len(permutations)):
        if kwargs.get("separate_labels_space"):
            target_offset = (pidx + 1) * 10
        dataset.append(DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                                       num_workers=kwargs.get("num_workers", 1),
                                       permutation=permutations[pidx], target_offset=target_offset,
                                       pad_to_32=kwargs.get("pad_to_32", False)))
    # For offline permuted we take the datasets and mix them.
    if kwargs.get("offline", False):
        train_sets = []
        test_sets = []
        for ds in dataset:
            train_sets.append(ds.train_set)
            test_sets.append(ds.test_set)
        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=kwargs.get("batch_size", 128), shuffle=True,
                                                   num_workers=kwargs.get("num_workers", 1), pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=kwargs.get("batch_size", 128), shuffle=False,
                                                  num_workers=kwargs.get("num_workers", 1), pin_memory=True)
        return [train_loader], [test_loader]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_padded_permuted_mnist(**kwargs):
    """
    Permuted MNIST dataset, padded to 32x32.
    """
    return ds_permuted_mnist(pad_to_32=True, **kwargs)


def ds_permuted_mnist_offline(**kwargs):
    """
    Permuted MNIST dataset. Offline means that all tasks are mixed together.
    """
    return ds_permuted_mnist(offline=True, **kwargs)


def ds_padded_permuted_mnist_offline(**kwargs):
    """
    Permuted MNIST dataset, padded to 32x32. Offline means that all tasks are mixed together.
    """
    return ds_permuted_mnist(pad_to_32=True, offline=True, **kwargs)


def ds_padded_cont_permuted_mnist(**kwargs):
    """
    Continuous Permuted MNIST dataset, padded to 32x32.
    :param num_epochs: Number of epochs for the training (since it builds distribution over iterations,
                            it needs this information in advance)
    :param iterations_per_virtual_epc: In continuous task-agnostic learning, the notion of epoch does not exists,
                                        since we cannot define 'passing over the whole dataset'. Therefore,
                                        we define "iterations_per_virtual_epc" -
                                        how many iterations consist a single epoch.
    :param contpermuted_beta: The proportion in which the tasks overlap. 4 means that 1/4 of a task duration will
                                consist of data from previous/next task. Larger values means less overlapping.
    :param permutations: The permutations which will be used (first task is always the original MNIST).
    :param batch_size: Batch size.
    :param num_workers: Num workers.
    """
    dataset = [DatasetsLoaders("CONTPERMUTEDPADDEDMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               total_iters=(kwargs.get("num_epochs")*kwargs.get("iterations_per_virtual_epc")),
                               contpermuted_beta=kwargs.get("contpermuted_beta"),
                               iterations_per_virtual_epc=kwargs.get("iterations_per_virtual_epc"),
                               all_permutation=kwargs.get("permutations", []))]
    test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
    train_loaders = [ds.train_loader for ds in dataset]

    return train_loaders, test_loaders


def ds_visionmix(**kwargs):
    """
    Vision mix dataset. Consists of: MNIST, notMNIST, FashionMNIST, SVHN and CIFAR10.
    """
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), pad_to_32=True),
               DatasetsLoaders("NOTMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("FashionMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("SVHN", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1))]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_cifar10and100(**kwargs):
    """
    CIFAR10 and CIFAR100 dataset. Consists of 6 tasks:
        1) CIFAR10
        2-6) Subsets of 10 classes from CIFAR100.
    """
    classes_lst = [[j for j in range(i * 10, (i + 1) * 10)] for i in range(0, 5)]
    dataset = [DatasetsLoaders("CIFAR100", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               reduce_classes=cl, preserve_label_space=False) for cl in classes_lst]
    dataset = [DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), preserve_label_space=False)] + dataset

    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_cifar10(**kwargs):
    """
    CIFAR10 dataset. No tasks.
    """
    dataset = [DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1))]

    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders