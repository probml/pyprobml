import superimport

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from skimage.transform import resize
import scipy.ndimage
from dataclasses import dataclass
import pyprobml_utils as pml

url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/binaryImages.csv'
df = pd.read_csv(url)
patterns = 2*df.to_numpy()[:, 1:]-1 


@dataclass
class Patterns:

  patterns: np.ndarray
  nimages: int
  
  @property
  def width(self):
    return int(self.patterns.shape[1]/self.nimages)
  
  @property
  def height(self):
    return self.patterns.shape[0]
  
  @property
  def shape(self):
    return self.patterns.shape
  
  def __getitem__(self, x):
    return self.patterns[x]
  
  def __setitem__(self, x, val):
    self.patterns[x] = val
  
  def copy(self):
    return Patterns(self.patterns.copy(), self.nimages)

def occlude_patterns(patterns, occulusion_rate=0.6):
  patterns_occluded = patterns.copy()
  for i in range(patterns.nimages):
    patterns_occluded[:, i*patterns.width:int(
        (i+occulusion_rate)*patterns.width)] = 1
  return patterns_occluded

def downsize_patterns(patterns, nsize):
  new_patterns = np.zeros((nsize, patterns.nimages*nsize))
  for i in range(patterns.nimages):
    new_patterns[:, i*nsize:(i+1)*nsize] = resize(
        patterns[:, i*patterns.width:int((i+1)*patterns.width)], 
        output_shape=(nsize, nsize), anti_aliasing=True)
  return Patterns(new_patterns, patterns.nimages)

def upsize_patterns(patterns, scale):
  nheight = int(np.ceil(scale*patterns.height))
  nwidth = int(np.ceil(scale*patterns.width))
  new_patterns = np.zeros((nheight, patterns.nimages*nwidth))
  for i in range(patterns.nimages):
    new_patterns[:, i*nwidth:(i+1)*nwidth] = scipy.ndimage.zoom(
        patterns[:, i*patterns.width:int((i+1)*patterns.width)],
        scale, order=0)
  return Patterns(new_patterns, patterns.nimages)

def convert_patterns_to_vectors(patterns):
  new_shape = patterns.width*patterns.height
  vectors = np.zeros((patterns.nimages, new_shape))
  for i in range(patterns.nimages):
    img = patterns.patterns[:, i*patterns.width:(i+1)*patterns.width]
    vectors[i] = img.reshape((new_shape, ))
  new_patterns = Patterns(vectors, patterns.nimages)
  new_patterns.og_width =  patterns.width
  new_patterns.og_height =  patterns.height
  return new_patterns

def convert_vectors_to_patterns(vectors):
  new_shape = vectors.og_width*vectors.nimages
  patterns = np.zeros((vectors.og_height, new_shape))
  for i in range(vectors.nimages):
    img = vectors[i, :]
    patterns[:, i*vectors.og_width:(i+1)*vectors.og_width] = img.reshape(
        (vectors.og_height, vectors.og_width))
  return Patterns(patterns, vectors.nimages)
  
def plt_patterns(patterns, ndisplay=None, figsize=30, name=None):

  assert patterns.nimages >= ndisplay, "number of images in the datset cannot \
  be less than number of images to be displayed"

  if not ndisplay:
    ndisplay=self.nimages
  fig, axs = plt.subplots(1, ndisplay, figsize=(figsize, figsize*ndisplay))
  fig.suptitle(f'{name}', fontsize=16, y=0.55)
  for i in range(ndisplay):
    axs[i].imshow(patterns[:, i*patterns.width:(i+1)*patterns.width], 
                  cmap="Greys")
  pml.savefig(f'{name}.pdf')
  plt.show()

class HopfieldNetwork(object):
  
  def fit(self, patterns, ntrained=3):
    r, c = patterns.shape
    W = np.zeros((c, c))
    for i in trange(ntrained):
      W += np.outer(patterns[i],patterns[i])
    W[np.diag_indices(c)] = 0
    self.W = W/r
  
  def recall(self, patterns, steps=2):
    sgn = np.vectorize(lambda x: -1 if x<0 else +1)
    for _ in trange(steps):
      patterns.patterns = sgn(np.dot(patterns.patterns, self.W))
    return patterns
  
  def hopfield_energy(self, patterns):
    return np.array([-0.5*np.dot(np.dot(p.T, self.W), p) for p in patterns])

pat = Patterns(patterns, 7)
plt_patterns(pat, 3, 10, name="hopfield_training")

pattern_vectors = convert_patterns_to_vectors(pat)
occluded_patterns = occlude_patterns(pat)
occluded_patterns_vectors = convert_patterns_to_vectors(occluded_patterns)

plt_patterns(occluded_patterns, 3, 10, 'hopfield_occluded')

net = HopfieldNetwork()
net.fit(pattern_vectors)
pattern_recovered = net.recall(occluded_patterns_vectors)

pattern_rec = convert_vectors_to_patterns(pattern_recovered)

plt_patterns(pattern_rec, 3, 10, 'hopfield_recall')
