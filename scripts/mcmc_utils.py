'''
Author : Ang Ming Liang
'''

import superimport

import numpy as np
#from tqdm.notebook import tqdm
from tqdm import tqdm

def slice_sample(init, dist, iters, sigma, burnin, step_out=True, rng=None):
    """
    based on http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/
    """

    # set up empty sample holder
    D = len(init)
    samples = np.zeros((D, iters))
    sigma = 5*np.ones(init.shape[-1])

    # initialize
    xx = init.copy()

    for i in tqdm(range(iters)):
        perm = list(range(D))
        rng.shuffle(perm)
        last_llh = dist(xx)

        for d in perm:
            llh0 = last_llh + np.log(rng.random())
            rr = rng.random(1)
            x_l = xx.copy()
            x_l[d] = x_l[d] - rr * sigma[d]
            x_r = xx.copy()
            x_r[d] = x_r[d] + (1 - rr) * sigma[d]

            if step_out:
                llh_l = dist(x_l)
                while llh_l > llh0:
                    x_l[d] = x_l[d] - sigma[d]
                    llh_l = dist(x_l)
                llh_r = dist(x_r)
                while llh_r > llh0:
                    x_r[d] = x_r[d] + sigma[d]
                    llh_r = dist(x_r)

            x_cur = xx.copy()
            while True:
                xd = rng.random() * (x_r[d] - x_l[d]) + x_l[d]
                x_cur[d] = xd.copy()
                last_llh = dist(x_cur)
                if last_llh > llh0:
                    xx[d] = xd.copy()
                    break
                elif xd > xx[d]:
                    x_r[d] = xd
                elif xd < xx[d]:
                    x_l[d] = xd
                else:
                    raise RuntimeError('Slice sampler shrank too far.')
        
        samples[:, i] = xx.copy().ravel()

    return samples[:, burnin:]
