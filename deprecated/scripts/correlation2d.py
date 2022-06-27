
import superimport

import numpy as np
from numpy import linalg as la
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

'''
https://en.wikipedia.org/wiki/Correlation_and_dependence#/media/File:Correlation_examples2.svg
The corr(X, Y) == 1 iff y = a * x + b

The following code demonstrate the above linked image that is included in
ML 1st Edition. There are two sliders in the UI that allow a user to change the
rotation and aspect ratio of the 2D point cloud. The corr(X, Y) value is also
displayed.

A line fitted by linear regression on the 2D point cloud is also shown. The SSE
of linear regression satisfies:
SSE / cov(Y,Y) = 1 - corr(X,Y)

Thus SSE == 0 iff corr(X,Y) == 1
'''

def Rotation(theta):
  ''' Return a 2D rotation matrix
  '''
  c = np.cos(theta)
  s = np.sin(theta)
  return np.array([[ c, -s],
                  [s, c]], dtype=np.float32)

def Scale(aspect):
  ''' Return a 2D scale matrix
  '''
  a = aspect
  return np.array([[1.0, 0.0],
                   [0.0, a]], dtype=np.float32)

def GeneratePoints(n, r):
  ''' Uniformlly sample n 2d points in the circle with radius r
  '''
  result = []
  while len(result) < n:
    p = r * (2.0 * np.random.random_sample() - 1.0),\
        r * (2.0 * np.random.random_sample() - 1.0)
    if la.norm(p, 2) <= r:
      result.append(p)

  return np.array(result, dtype=np.float32)


def LinearRegressionOn2DPoints(points):

  points_T = np.transpose(points)
  X0, Y0 = points_T[0, :], points_T[1, :]

  n = len(X0)
  ones = np.ones(n)
  A = np.vstack([X0, ones]).T
  a, b = la.lstsq(A, Y0)[0]

  alpha = np.array([a, b]).T
  Y = np.dot(np.vstack((X0, np.ones(len(X0)))).T, alpha)

  return X0, Y0, Y

def Correlation2DPoints(points):
  X = points[:, 0]
  Y = points[:, 1]
  C = np.corrcoef(points[:, 0], points[:, 1])
  return C[0,1]

def TransformPoints(params):
  R = Rotation(params["theta"])
  S = Scale(params["aspect"])
  T = np.dot(R, S)
  points = params["original_points"]
  params["points"] = np.array([np.dot(T, np.transpose(point)) for point in points], dtype=np.float32)


def updateHandler(key, text, points_plot, line_plot, params):
  def update(v):
    params[key] = v
    TransformPoints(params)
    X0, Y0, Y = LinearRegressionOn2DPoints(params["points"])
    points_plot.set_xdata(X0)
    points_plot.set_ydata(Y0)
    line_plot.set_xdata(X0)
    line_plot.set_ydata(Y)
    text.set_text("Corr(x,y) %f" % Correlation2DPoints(params["points"]))
  return update


def main():
  points = GeneratePoints(1000, 4.0)
  theta = np.pi / 4.0
  aspect = 0.5
  params = {"theta": theta,
            "aspect": aspect,
            "original_points": points,
            "points": points}

  TransformPoints(params)
  X0, Y0, Y = LinearRegressionOn2DPoints(params["points"])

  points_plot, = plt.plot(X0, Y0, 'o')
  line_plot, = plt.plot(X0, Y, 'r')
  plt.axis('equal')

  # Add UI Text and Sliders
  text = plt.text(-4.5, 3.5, "Corr(x,y) %f" % Correlation2DPoints(params["points"]), fontsize=15)
  ax_aspect = plt.axes([0.25, 0.1, 0.65, 0.03])
  ax_theta = plt.axes([0.25, 0.15, 0.65, 0.03])
  aspect_slider = Slider(ax_aspect, 'Aspect', 0.0, 1.0, valinit=aspect)
  theta_slider = Slider(ax_theta, 'Theta', -0.5 * np.pi, 0.5 * np.pi, valinit=theta)
  aspect_slider.on_changed(updateHandler("aspect", text, points_plot, line_plot, params))
  theta_slider.on_changed(updateHandler("theta", text, points_plot, line_plot, params))

  plt.show()

if __name__ == "__main__":
    main()