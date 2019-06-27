import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


x = np.linspace(0,1,10000)

H = lambda x: -(x*np.log2(x) + (1-x) * np.log2(1-x))

plt.plot(x, H(x), '-b', lw=3)
plt.xlim((-0.01, 1.01))
plt.ylim((0, 1.01))

plt.xlabel("p(X = 1)")
plt.ylabel("H(X)")

ticks = [0, 0.5, 1]
plt.xticks(ticks)
plt.yticks(ticks)

save_fig("bernoulliEntropy.pdf")
plt.show()
