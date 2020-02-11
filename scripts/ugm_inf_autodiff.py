# inference for UGM by autodiff
# we consider a length 3 HMM with K states
# h1 - h2 - h3

import numpy as onp # original numpy
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update
from functools import partial

K = 2
nnodes = 3
edges = [ [0,1], [1,2]]
nedges = len(edges)
onp.random.seed(0)
nodePots = []
nodePotsLog = []
for i in range(nnodes):
    nodePot = onp.random.rand(K)
    nodePots.append(nodePot)
    nodePotsLog.append(onp.log(nodePot))
edgePots = []
edgePotsLog = []
for i in range(nedges):
    edgePot = onp.random.rand(K,K)
    edgePots.append(edgePot)
    edgePotsLog.append(onp.log(edgePot))
    
def compute_joint(nodePots, edgePots):    
    joint = onp.ones((K, K, K))
    np0 = nodePots[0].reshape((K,1,1))
    joint = joint * np0
    np1 = nodePots[1].reshape((1,K,1))
    joint = joint * np1
    np2 = nodePots[2].reshape((1,1,K))
    joint = joint * np2
    ep0 = edgePots[0].reshape((K,K,1))
    joint = joint * ep0
    ep1 = edgePots[1].reshape((1,K,K))
    joint = joint * ep1
    return joint

joint = compute_joint(nodePots, edgePots)

'''
# sanity check
x = [0,1,0]
p = 1
for i in range(nnodes):
    p = p * nodePots[i][x[i]]
for i, e in enumerate(edges):
    src = e[0]
    sink = e[1]
    p = p * edgePots[i][x[src], x[sink]] 
j = joint[x[0], x[1], x[2]]
assert p == j
#assert(joint[0,1,0] == nodePots[0][0] * nodePots[1][1] * nodePots[2][0])
'''

# Compute marginals by brute force
Z = onp.sum(joint)
jointNorm = joint / Z
marg0 = onp.sum(jointNorm, (1, 2))
marg1 = onp.sum(jointNorm, (0, 2))
marg2 = onp.sum(jointNorm, (0, 1))

# Now use autodiff!

def compute_Z(nodePots):
    factors = [nodePots[0], nodePots[1], nodePots[2], edgePots[0], edgePots[1]]
    str = 'A,B,C, AB,BC'
    Z = jnp.einsum(str, *factors)
    return Z

ZZ = onp.float(compute_Z(nodePots))
assert onp.isclose(Z, ZZ)

# Computes A(eta), where A=logZ and eta=logPot
def compute_logZ(nodePotsLog):
    nodePots = []
    for i in range(len(nodePotsLog)):
        nodePots.append(jnp.exp(nodePotsLog[i]))
    return jnp.log(compute_Z(nodePots))

# sanity check


g = grad(compute_logZ)(nodePotsLog)
margProbs = []
for i in range(nnodes):
    margProbs.append(onp.array(g[i]))



assert onp.allclose(margProbs[0], marg0)
assert onp.allclose(margProbs[1], marg1)
assert onp.allclose(margProbs[2], marg2)



