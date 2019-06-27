import numpy as np
import matplotlib.pyplot as plt 
import os
from pyprobml_utils import save_fig, get_data_dir
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

data_dir = get_data_dir()
img = mpimg.imread(os.path.join(data_dir, "clown.png"))

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
    save_fig("svdImageDemoClown{}.pdf".format(k))
    plt.show()

k = 100
plt.plot(np.log(sigma[:k]), 'r-', linewidth=4, label="Original")
plt.ylabel(r"$log(\sigma_i)$")
plt.xlabel("i")

# permutation only permutes the rows, which does not destroy the structure
# The singular values are identical
x2 = np.random.permutation(X)
# so we convert to a 1d vector, permute, and convert back
x1d = X.ravel()
np.random.shuffle(x1d) # inplace
x2 = x1d.reshape(X.shape)
U, sigma2, V = np.linalg.svd(x2, full_matrices = False)
plt.plot(np.log(sigma2[:k]), 'g:', linewidth=4, label="Randomized")
plt.legend()
save_fig("svdImageDemoClownSigmaScrambled.pdf")
plt.show()
