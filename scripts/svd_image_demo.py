import superimport

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import os
import pyprobml_utils as pml

from PIL import Image
import requests
import io


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

r = requests.get('https://github.com/probml/probml-data/blob/main/data/clown.png?raw=true', stream=True)
img = Image.open(io.BytesIO(r.content))
#img.save('clown.png')
#img = matplotlib.image.imread("clown.png")
#X = rgb2gray(img)
X = np.array(img)

r = np.linalg.matrix_rank(X)
print(r)

U, sigma, V = np.linalg.svd(X, full_matrices=True)
ranks = [1, 2, 5, 10, 20, r]
R = len(ranks)

for i in range(R):
    k = ranks[i]
    #x_hat = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])  
    x_hat = np.dot(np.dot(U[:, :k], np.diag(sigma[:k])), V[:k, :])  
    plt.imshow(x_hat, cmap='gray')
    plt.title("rank {}".format(k))
    plt.axis("off")
    pml.savefig("svdImageDemoClown{}.pdf".format(k))
    plt.show()

k = 100
plt.plot(np.log(sigma[:k]), 'r-', linewidth=4, label="Original")
plt.ylabel(r"$log(\sigma_i)$")
plt.xlabel("i")


# Compare this to a random shuffled version of the image
x2 = np.random.permutation(X)
# so we convert to a 1d vector, permute, and convert back
x1d = X.ravel()
np.random.shuffle(x1d) # inplace
x2 = x1d.reshape(X.shape)
U, sigma2, V = np.linalg.svd(x2, full_matrices = False)
plt.plot(np.log(sigma2[:k]), 'b', linewidth=4, label="Randomized")
plt.legend()
pml.savefig("svdImageDemoClownSigmaScrambled.pdf")
plt.show()
