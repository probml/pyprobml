#https://stackoverflow.com/questions/30849910/matplotlib-heatmap-image-rotated-when-heatmap-plot-over-it
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import urllib
import scipy

#downloading an example image
url = "http://tekeye.biz/wp-content/uploads/2013/01/small_playing_cards.png"
#urllib.urlretrieve(url, "/tmp/cards.png") # python2
urllib.request.urlretrieve(url, "/tmp/cards.png") #python3

#reading and plotting the image
im = plt.imread('/tmp/cards.png')
im.shape #281, 820, 4

if False:
    x=numpy.random.normal(500, 100, size=1000)
    y=numpy.random.normal(100, 50, size=1000)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    plt.imshow(heatmap, extent=extent,alpha=.5,origin='upper')

plt.figure()
#heatmap = np.random.rand(im.shape[0], im.shape[1])
heatmap = np.random.rand(100,100)
heatmap = scipy.misc.imresize(heatmap, im.shape[0:2])
plt.imshow(heatmap)
#plt.axis('off')
plt.show()

plt.figure()
implot = plt.imshow(im, origin='upper')
plt.imshow(heatmap, alpha=.5, origin='upper')
#plt.axis('off')
plt.show()
