# splines in 1d
# We use the cherry blossom daa from sec 4.5 of "Statistical Rethinking"
# We use temperature as the target variable, to match a draft version of the book,
# https://github.com/Booleans/statistical-rethinking/blob/master/Statistical%20Rethinking%202nd%20Edition.pdf
# The published version uses  day of year as target, which is less visually interesting.
# This an MLE version of the Bayesian numpyro code from 
# https://fehiepsi.github.io/rethinking-numpyro/04-geocentric-models.html

import superimport

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

from scipy.interpolate import BSpline

from scipy import stats
from patsy import bs, dmatrix

import sklearn
from sklearn.linear_model import LinearRegression, Ridge



#https://stackoverflow.com/questions/61807542/generate-a-b-spline-basis-in-scipy-like-bs-in-r


def make_splines_scipy(x, num_knots, degree=3):
  knot_list = np.quantile(x, q=np.linspace(0, 1, num=num_knots))
  knots = np.pad(knot_list, (3, 3), mode="edge")
  B = BSpline(knots, np.identity(num_knots + 2), k=degree)(x)
  # according to scipy documentation
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
  # if degree = k, ncoef = n,  nknots = n + k + 1
  # so if k=3, ncoef = nknots - 4
  # where nknots = num_knot + 6 (because of 3 pad on left, 3 on right)
  # so ncoef= num_knots + 6 - 4 = num_knots + 2
  return B


def make_splines_patsy(x, num_knots, degree=3):
  knot_list = np.quantile(x, q=np.linspace(0, 1, num=num_knots))
  #B  = bs(x, knots=knot_list, degree=degree)  # ncoef = knots + degree + 1 
  B  = bs(x, df=num_knots, degree=degree)   # uses quantiles
  return B


def plot_basis(x, B, w=None):
  if w is None: w = np.ones((B.shape[1]))
  fig, ax = plt.subplots()
  ax.set_xlim(np.min(x), np.max(x))
  for i in range(B.shape[1]):
      ax.plot(x, (w[i] * B[:, i]), "k", alpha=0.5)
  return ax

def plot_basis_with_vertical_line(x, B, xstar):
  ax = plot_basis(x, B)
  num_knots = B.shape[1]
  ndx = np.where(x==xstar)[0][0]
  for i in range(num_knots):
    yy = B[ndx,i]
    if yy>0:
      ax.scatter(xstar, yy, s=40)
  ax.axvline(x=xstar)
  return ax

def plot_pred(mu, x, y):
  plt.figure()
  plt.scatter(x, y, alpha=0.5)
  plt.plot(x, mu, 'k-', linewidth=4)

  
def main():
    url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/cherry_blossoms.csv'
    cherry_blossoms = pd.read_csv(url, sep=';')
    df = cherry_blossoms
    
    display(df.sample(n=5, random_state=1))
    display(df.describe())
    
    
    df2 = df[df.temp.notna()]  # complete cases 
    x = df2.year.values.astype(float)
    y = df2.temp.values.astype(float)
    xlabel = 'year'
    ylabel = 'temp'
  
    nknots = 15
    
    #B =  make_splines_scipy(x, nknots)
    B =  make_splines_patsy(x, nknots)
    print(B.shape)
    plot_basis_with_vertical_line(x, B, 1200)
    plt.tight_layout()
    plt.savefig(f'../figures/splines_basis_vertical_MLE_{nknots}_{ylabel}.pdf', dpi=300)
    
    
    #reg = LinearRegression().fit(B, y)
    reg = Ridge().fit(B, y)
    w = reg.coef_
    a = reg.intercept_
    print(w)
    print(a)
    
    plot_basis(x, B, w)
    plt.tight_layout()
    plt.savefig(f'../figures/splines_basis_weighted_MLE_{nknots}_{ylabel}.pdf', dpi=300)
    
    mu = a + B @ w
    plot_pred(mu, x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'../figures/splines_point_pred_MLE_{nknots}_{ylabel}.pdf', dpi=300)
      

    
main()


    
    