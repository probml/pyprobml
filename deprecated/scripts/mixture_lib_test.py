# Test for mixture_general_lib
# Author : Aleyna Kara(@karalleyna)

import superimport

import jax.numpy as jnp
import distrax
from mixture_lib import MixtureSameFamily

from jax import test_util as jtu
from absl.testing import absltest


class TestMixtureSameFamily(jtu.JaxTestCase):

    def test_posterior_marginal(self):
        mix_dist_probs = jnp.array([0.1, 0.9])
        component_dist_probs = jnp.array([[.2, .3, .5],
                                          [.7, .2, .1]])
        bm = MixtureSameFamily(
            mixture_distribution=distrax.Categorical(
                probs=mix_dist_probs),
            components_distribution=distrax.Categorical(
                probs=component_dist_probs))

        marginal_dist = bm.posterior_marginal(jnp.array([0., 1., 2.]))
        marginals = marginal_dist.probs

        self.assertEqual((3, 2), marginals.shape)

        expected_marginals = jnp.array([
            [(.1 * .2) / (.1 * .2 + .9 * .7), (.9 * .7) / (.1 * .2 + .9 * .7)],
            [(.1 * .3) / (.1 * .3 + .9 * .2), (.9 * .2) / (.1 * .3 + .9 * .2)],
            [(.1 * .5) / (.1 * .5 + .9 * .1), (.9 * .1) / (.1 * .5 + .9 * .1)]
        ])

        self.assertAllClose(marginals, expected_marginals)

    def test_posterior_mode(self):
        mix_dist_probs = jnp.array([[0.5, 0.5],
                                    [0.01, 0.99]])
        locs = jnp.array([[-1., 1.],
                          [-1., 1.]])
        scale = jnp.array([1.])

        gm = MixtureSameFamily(
            mixture_distribution=distrax.Categorical(
                probs=mix_dist_probs),
            components_distribution=distrax.Normal(
                loc=locs,
                scale=scale))

        mode = gm.posterior_mode(jnp.array([[1.], [-1.], [-6.]]))

        self.assertEqual((3, 2), mode.shape)
        self.assertAllClose(jnp.array([[1, 1], [0, 1], [0, 0]]), mode)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())