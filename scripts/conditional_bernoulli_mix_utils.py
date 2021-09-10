# Necessary functions for demo and ClassConditionalBMM
# Author : Aleyna Kara(@karalleyna)

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split

import torchvision

import numpy as np


def encode(word, L, type_):
    # Encodes the word into list of integers
    arr = np.array([], dtype=int)
    for c in word:
        if c == '-':
            arr = np.append(arr, [L - 1])
        elif type_ == 'all':
            arr = np.append(arr,
                            [ord(c) - 39 if c.isupper() else ord(c) - 97])  # ascii no of A : 65, ascii no of a : 97
        elif type_ == "upper":
            arr = np.append(arr, [ord(c.upper()) - 65])
        elif type_ == "lower":
            arr = np.append(arr, [ord(c.lower()) - 65])
    return arr


def decode(idx, L, type_):
    # Converts the list of integers into a word
    return jnp.where(idx == L - 1, ord("-"),
                     jnp.where((type_ == "all") & (idx >= L // 2), idx - L // 2 + 65,
                               jnp.where(type_ == "upper", idx + 65, idx + 97))).flatten()


def fake_test_data(test_words, dataset, targets, L, type_, rng_key=None):
    """
    Creates the test images for given test words via the given dataset and its class labels.

    Parameters
    ----------
    test_words: list
        Collection of words to be tested
    dataset : array
        Dataset
    targets : array
        The ground-truth labels of the dataset
    L : int
        Total number of different characters and blank character
    type : str
      "all" : Includes both uppercase and lowercase letters
      "lower" : Includes only lowercase letters
      "upper" : Includes only uppercase letters
    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array
        Test images
    """
    dataset_with_blank = jnp.vstack([dataset, jnp.zeros((1, 28, 28))])
    targets_with_blank = jnp.append(targets, L - 1)
    n_img = dataset_with_blank.shape[0]
    test_images = []
    if rng_key is None:
        rng_key = PRNGKey(0)

    keys = [dev_arr for dev_arr in split(rng_key, len(test_words))]
    for word, key in zip(test_words, keys):
        classes = encode(word, L, type_)
        keys_ = [dev_array for dev_array in split(key, len(word))]

        get_img_index = lambda key, target: jax.random.choice(key, jnp.where(targets_with_blank == target)[0])

        img_indices = jax.tree_multimap(get_img_index, keys_, classes.tolist())
        test_images.append(dataset_with_blank[img_indices, ...])
    return test_images


def get_emnist_images_per_class(select_n):
    """
    Gets training data with its targets

    Parameters
    ----------
    select_n : int
        Number of data points belonging to one class of the dataset

    Returns
    -------
    * array
        Dataset
    * array
        The ground-truth labels of the dataset
    """
    dataset = torchvision.datasets.EMNIST(root=".", split="byclass", download=True)
    targets = dataset.targets.numpy() - 10
    dataset = dataset.data.permute(0, 2, 1).numpy()
    dataset = dataset[np.argwhere(targets >= 0)]
    targets = targets[np.argwhere(targets >= 0)].flatten()

    target_ls = []
    for i in range(52):
        target_ls.append(np.argwhere(targets == i)[:select_n])

    targets = targets[np.concatenate(target_ls)].flatten()
    dataset = dataset[np.concatenate(target_ls)].squeeze()

    perm = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(dataset.shape[0]))
    dataset, targets = dataset[perm], targets[perm]
    dataset = (dataset > 0.5).astype(int)
    return dataset, targets


def get_decoded_samples(words):
    """
    Gets the string representations of the words each of which is shown by a row whose values are the ASCII indices

    Parameters
    ----------
    words: array
        A matrix in which all words represented by a row attained by the decode function

    Returns
    -------
    * array
        Char array including each original word as a string
    """
    # list(map(lambda x: "".join(map(chr, x)), words))
    # np.array(words, dtype=np.int32).view("U4")
    return np.apply_along_axis(lambda x: "".join(list(x)), 1, np.vectorize(chr)(words.astype(int)))