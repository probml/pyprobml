from ..plot import MatplotlibMixin
from .. import calculate


class BayesPredictor(MatplotlibMixin):

    def __init__(self, observations_space):
        self.observations_space = observations_space
        self.filename_template = f'healthyLevels{self.__class__.__name__}{{}}uninfPrior.pdf'
        self.title_template = 'Bayes predictive, n={}, uninfPrior'

    def calc_posterior_predictive(self, observations, num_observations):
        # Predictive distribution: Tenenbaum thesis eqn 3.16.
        x, y, width, height = calculate.smallest_rectangle_enclosing(observations)
        x_diffs  = calculate.diffs(self.observations_space[:, 0], x, width)
        y_diffs  = calculate.diffs(self.observations_space[:, 1], y, height)
        return calculate.proba_from_diffs(x_diffs, y_diffs, width, height, num_observations)

    def fit(self, observations):
        self.title = self.title_template.format(len(observations))
        self.filename = self.filename_template.format(len(observations))
        self.posterior_predictive = self.calc_posterior_predictive(observations, len(observations))
        self.observations = observations
       
    def _plot(self):
        self.plot_points(self.observations)
        self.plot_contours(self.observations_space, self.posterior_predictive)

