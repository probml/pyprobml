# adpated from http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.correlate2d.html

import superimport

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc

print("cross correlation demo")
face = misc.face() - misc.face().mean()
face = face.sum(-1)
template = np.copy(face[700:800, 310:380]) # right eye
template -= template.mean()
noisyface = face + np.random.randn(*face.shape) * 50 # add noise
corr = signal.correlate2d(noisyface, template, boundary='symm', mode='same')
y, x = np.unravel_index(-1*np.argmax(corr), corr.shape) # find the match


fig, ((ax_orig, ax_template), (ax_noisy, ax_corr)) = plt.subplots(2, 2)

ax_orig.imshow(face, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_orig.plot(x, y, 'ro')

ax_template.imshow(template, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()


ax_noisy.imshow(noisyface, cmap='gray')
ax_noisy.set_title('Noisy')
ax_noisy.set_axis_off()
ax_noisy.plot(x, y, 'ro')


ax_corr.imshow(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()

fig.show()
