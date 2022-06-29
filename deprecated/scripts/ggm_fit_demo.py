# This file show a demo of showing that MLE(precision matrix) of a ggm
# satisfies the constraints mentioned in the GGM section of the book.

import superimport

import numpy as np
from ggm_fit_htf import ggm_fit_htf

G = np.array([0., 1., 0., 1,
              1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0]).reshape((4, 4))

S = np.array([10., 1., 5., 4.,
              1., 10., 2., 6.,
              5., 2., 10., 3.,
              4., 6., 3., 10]).reshape((4, 4))

max_iter = 30


prec_mat = ggm_fit_htf(S, G, max_iter)
sigma = np.linalg.inv(prec_mat)

guide_sigma = np.array([10., 1., 1.31, 4,
                        1., 10., 2., 0.87,
                        1.31, 2., 10., 3,
                        4., 0.87, 3., 10.]).reshape(4, 4)

guide_prec_mat = np.array([0.12, -0.01, 0, -0.05,
                           -0.01, 0.11, -0.02, 0.,
                           0, -0.02, 0.11, -0.03,
                           -0.05, 0, -0.03, 0.13]).reshape(4, 4)

assert np.all(sigma - guide_sigma < 1e-2)
assert np.all(prec_mat - guide_prec_mat < 1e-2)
