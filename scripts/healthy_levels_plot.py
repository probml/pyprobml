# Based on https://github.com/probml/pmtk3/blob/master/demos/healthyLevels.m
# Converted by John Fearns - jdf22@infradead.org
# Josh Tenenbaum's Healthy Levels game

import superimport

import numpy as np
import matplotlib.pyplot as plt
#from pyprobml_utils import save_fig

# Ensure stochastic reproducibility.
np.random.seed(11)

# Generate the synthetic data - positive examples only. Data is returned
# as a 2 column table. First column is colesterol levels, second is insulin.
def generate_data():
    # Healthy levels we are trying to discover
    c_low = 0.35; c_high = 0.55;
    i_low = 0.45; i_high = 0.65;
        
    # Cheat and use interesting-looking data.
    c =  [0.351, 0.363, 0.40, 0.54, 0.45, 0.49, 0.48, 0.50, 0.45, 0.41, 0.53, 0.54]
    i =  [0.452, 0.64, 0.46, 0.55, 0.55, 0.50, 0.49, 0.61, 0.58, 0.46, 0.53, 0.64]
    return np.column_stack([c, i])

# Calculates the range of the provided data points: x_min, x_max, y_min, y_max, x_scale, y_scale.
def calc_data_range(data):
    x_min = np.min(data[:,0], 0)
    x_max = np.max(data[:,0], 0)
    y_min = np.min(data[:,1], 0)
    y_max = np.max(data[:,1], 0)
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    
    return x_min, x_max, y_min, y_max, x_scale, y_scale

# Returns a matrix. The rows are the hypotheses,
# the first column is the x-scale value, the second column is
# the y-scale value.
def get_hypotheses():
    stepsize = 0.01
    x = np.arange(stepsize, 1, stepsize)
    xs, ys = np.meshgrid(x, x)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    return np.column_stack([xs, ys])

# Returns an unnormalised prior on the hypotheses that is
# proportional to the inverse product of the hypothesis x- and y-scale
def get_uninformative_prior(hypotheses):
    s1 = hypotheses[:,0]
    s2 = hypotheses[:,1]
    return 1 / (s1 * s2)

def calc_likelihood(hypotheses, data):
    n = data.shape[0]
    s1 = hypotheses[:,0]
    s2 = hypotheses[:,1]
    c_min, c_max, i_min, i_max, c_scale, i_scale = calc_data_range(data)
    
    # Truth values indicating whether each hypothesis supports the data (hypothesis scale must
    # cover the data range)
    indications = np.logical_and(s1 > c_scale, s2 > i_scale)

    # Likelihood is proportional to 1/scale in each direction for each data point,
    # or to zero if the hypothesis doesn't support the data.
    return indications / np.power(s1*s2, n)
    
def calc_posterior(likelihood, prior):
    unnormalised = likelihood * prior
    return unnormalised / np.sum(unnormalised)

# Plot maximum likelihood based predictions for the top 3 and 12 data points.
def plot_ml(data):
    top_n = [3, 12];
    for i in range(len(top_n)):
        n = top_n[i]
        figure = plt.figure()
        plot_data = data[:n]
        # Plot the data and the smallest rectangle enclosing it.
        c_min, c_max, i_min, i_max, c_scale, i_scale = calc_data_range(plot_data)
        plt.gca().add_patch(
                plt.Rectangle((c_min, i_min),
                              c_scale, i_scale, fill=False,
                              edgecolor='black', linewidth=3)
                )
        plt.scatter(plot_data[:,0], plot_data[:,1], marker='+', color='red', zorder=10, linewidth=3)
        plt.title('MLE predictive, n={}'.format(n), fontsize=12, y=1.03)
        plt.axis('square')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        
        filename = '../figures/healthyLevelsMLPred{}.pdf'.format(n)
        plt.savefig(filename)
        plt.show(block=False)

def plot_posterior_samples(data, hypotheses, prior):
    top_n = [3, 12]
    for i in range(len(top_n)):
        plot_data = data[:top_n[i]]
        plot_lik = calc_likelihood(hypotheses, plot_data)
        plot_post = calc_posterior(plot_lik, prior)
        
        figure = plt.figure()
        prior_type = 'uninfPrior'
        title = r'samples from $p(h|D_{{1:{}}})$, {}'.format(top_n[i], prior_type)
        plot_sampled_hypotheses(hypotheses, plot_post, plot_data, title)
        filename = '../figures/healthyLevelsSamples{}{}.pdf'.format(top_n[i], prior_type)
        plt.title(title, fontsize=12, y=1.03)
        plt.savefig(filename)
        plt.show(block=False)

# Returns greyscale colours that reflect the supplied relative probabilities,
# ranging from black for the most probable, to light grey for the least probable.
def colours(probabilities):
    max_prob = np.max(probabilities)
    intensities = 1 - (0.25 + 0.75 * probabilities / max_prob)
    intensities = intensities.reshape(intensities.shape + (1,))
    
    # Repeat the same intensity for all RGB channels.
    return np.repeat(intensities, 3, intensities.ndim - 1)
    
def plot_sampled_hypotheses(hypotheses, posterior, data, title):
    # Take 10 samples from the posterior.
    N = 10
    samples_index = np.random.choice(len(posterior), N, p=posterior)
    
    # Reorder the samples from least to most probable, to help later with
    # drawing order so that darker, higher probability samples are drawn on top of
    # lighter, lower probability samples.
    samples_prob = posterior[samples_index]
    samples_index = samples_index[np.argsort(samples_prob)]
    del samples_prob
    
    samples_s1 = hypotheses[samples_index, 0]
    samples_s2 = hypotheses[samples_index, 1]
    plt.scatter(data[:,0], data[:,1], marker='+', color='red', zorder=10, linewidth=3)
    
    c_min, c_max, i_min, i_max, c_scale, i_scale = calc_data_range(data)
    samples_left = c_min - (samples_s1 - c_scale) / 2
    samples_lower = i_min - (samples_s2 - i_scale) / 2
    samples_colour = colours(posterior[samples_index])
    
    for i in range(N):
        plt.gca().add_patch(
                plt.Rectangle((samples_left[i], samples_lower[i]),
                              samples_s1[i], samples_s2[i], fill=False,
                              edgecolor=samples_colour[i], linewidth=3)
                )
    plt.xlim(0.2, 0.7)
    plt.ylim(0.3, 0.8)
        
def plot_bayes(data):
    top_n = [3, 12]
    for i in range(len(top_n)):
        plot_data = data[:top_n[i]]
        plot_contour(plot_data, i == len(top_n)-1)
        
def plot_contour(data, is_last_plot):
    # Prepare plot x-y points in various shapes.
    n = data.shape[0]
    stepsize = 0.01
    x = np.arange(0.00, 1.0, stepsize) + 0.01
    y = x
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        
    # Predictive distribution: Tenenbaum thesis eqn 3.16.
    d1, d2, r1, r2 = neighbour(data, points)
    denom = (1 + (d1/r1)) * (1 + (d2/r2))
    p = np.power(1/denom, n-1)
    p = p / np.sum(p)
    
    # Prepare for plotting
    pp = p.reshape(xx.shape)
    
    # Plot the predictive contours and data
    figure = plt.figure()
    plt.gray()
    plt.contour(xx, yy, pp)
    plt.scatter(data[:,0], data[:,1], marker='+', color='red', zorder=10, linewidth=3)
    
    plt.title('Bayes predictive, n={}, uninfPrior'.format(n), fontsize=12, y=1.03)
    plt.axis('square')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    filename = '../figures/healthyLevelsBayesPred{}UninfPrior.pdf'.format(n)
    plt.savefig(filename)
    plt.show(block=is_last_plot)
    
def neighbour(data, points):
    # Calculate d1, d2 of the points from the data. d_(j)[i] is 0 whenever points[i,j-1] is
    # within the span of the data in the jth dimension, otherwise it's the distance to the
    # nearest neighbour along that dimension.
    data1_min, data1_max, data2_min, data2_max, data1_scale, data2_scale = calc_data_range(data)
    d1 = (points[:,0] < data1_min) * abs(points[:,0] - data1_min) + (points[:,0] > data1_max) * abs(data1_max - points[:,0])
    d2 = (points[:,1] < data2_min) * abs(points[:,1] - data2_min) + (points[:,1] > data2_max) * abs(data2_max - points[:,1])
    return d1, d2, data1_scale, data2_scale

def main():
    data = generate_data()
    hypotheses = get_hypotheses()
    prior = get_uninformative_prior(hypotheses)

    plot_ml(data)
    plot_posterior_samples(data, hypotheses, prior)
    plot_bayes(data)

main()
