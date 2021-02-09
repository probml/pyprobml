import numpy as np

def smallest_rectangle_enclosing(observations):
    x_min = np.min(observations[:,0], 0)
    x_max = np.max(observations[:,0], 0)
    y_min = np.min(observations[:,1], 0)
    y_max = np.max(observations[:,1], 0)
    x_span = x_max - x_min
    y_span = y_max - y_min
    return x_min, y_min, x_span, y_span

# Returns an unnormalised prior on the hypotheses that is
# proportional to the inverse product of the hypothesis x- and y-scale
def uninformative_prior(hypotheses):
    x_span_hyp = hypotheses[:,0]
    y_span_hyp = hypotheses[:,1]
    return 1 / x_span_hyp*y_span_hyp


def likelihood(hypotheses, observations):
    n = len(observations)
    x_span_hyp = hypotheses[:,0]
    y_span_hyp = hypotheses[:,1]
    _, _, x_span, y_span = smallest_rectangle_enclosing(observations)
    
    # Truth values indicating whether each hypothesis supports the obbservations (hypothesis scale must
    # cover the obbservations range)
    indications = np.logical_and(x_span_hyp > x_span, y_span_hyp > y_span)

    # Likelihood is proportional to 1/scale in each direction for each obbservations point,
    # or to zero if the hypothesis doesn't support the obbservations.
    return indications / np.power(x_span_hyp*y_span_hyp, n)
    
def posterior(likelihood, prior):
    unnormalised = likelihood * prior
    return unnormalised / np.sum(unnormalised)

       
def diffs(points, min_, span):
    # is 0 whenever the point is
    # within the span of the observation, otherwise it's the distance to the
    # nearest neighbour along that dimension.
    max_ = min_ + span
    return (points < min_) * abs(points - min_) + (points > max_) * abs(max_ - points)


def proba_from_diffs(x_diffs, y_diffs, width, height, n_observations):
    p_denom = (1 + (x_diffs/width)) * (1 + (y_diffs/height))
    p = np.power(1/p_denom, n_observations-1)
    r = p / np.sum(p)
    return r


def draw_from_posterior(hypotheses, posterior, size):
    samples_index = np.random.choice(len(posterior), size, p=posterior)

    # Reorder the samples from least to most probable, to help later with
    # drawing order so that darker, higher probability samples are drawn on top of
    # lighter, lower probability samples.
    samples_prob = posterior[samples_index]
    samples_index = samples_index[np.argsort(samples_prob)]
    
    samples_posterior = posterior[samples_index]
    samples_width, samples_height = hypotheses[samples_index, 0], hypotheses[samples_index, 1]
    
    return  samples_width, samples_height, samples_posterior
