import numpy as np
import matplotlib.pyplot as plt 
import os
from pyprobml_utils import save_fig
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread("../data/clown.png")

X = rgb2gray(img)    

r = np.linalg.matrix_rank(X)
print(r)

U, sigma, V = np.linalg.svd(X, full_matrices=True)
ranks = [1, 2, 5, 10, 20, r]
R = len(ranks)

for i in range(R):
    k = ranks[i]
    x_hat = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])    
    plt.imshow(x_hat, cmap='gray')
    plt.title("rank {}".format(k))
    plt.axis("off")
    plt.show()
    save_fig("svdImageDemoClown{}.pdf".format(k))

k = 100
plt.plot(np.log(sigma[:k]), 'r-', linewidth=4, label="Original")
plt.ylabel(r"$log(\sigma_i)$")
plt.xlabel("i")

x2 = np.random.permutation(X)
U, sigma2, V = np.linalg.svd(x2, full_matrices = False)
plt.plot(np.log(sigma2[:k]), 'g:', linewidth=4, label="Randomized")
plt.legend()
plt.show()
save_fig("svdImageDemoClownSigmaScrambled.pdf")
