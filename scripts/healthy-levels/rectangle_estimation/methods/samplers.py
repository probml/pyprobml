from ..plot import MatplotlibMixin
from .. import calculate


class Sampler(MatplotlibMixin):

    def __init__(self, hypotheses, size):
        self.hypotheses = hypotheses
        self.sample_size = size
        self.prior = calculate.uninformative_prior(hypotheses)
        self.filename_template = f'healthyLevels{self.__class__.__name__}{{}}uninfPrior.pdf'
        self.title_template = r'samples from $p(h|D_{{1:{}}})$, uninfPrior'

    def fit(self, observations):
        self.title = self.title_template.format(len(observations))
        self.filename = self.filename_template.format(len(observations))
        
        likelihood = calculate.likelihood(self.hypotheses, observations)
        posterior = calculate.posterior(likelihood, self.prior)
        self.samples_width, self.samples_height, self.samples_posterior = calculate.draw_from_posterior(self.hypotheses, posterior, self.sample_size)
        
        x, y, width, height = calculate.smallest_rectangle_enclosing(observations)
        self.samples_x = x - (self.samples_width - width) / 2
        self.samples_y = y - (self.samples_height - height) / 2
        self.observations = observations


    def _plot(self):
        self.plot_points(self.observations)
        colors = self.colors_from_proba(self.samples_posterior)
        for index in range(self.sample_size):
            rectangle = (
                self.samples_x[index],
                self.samples_y[index],
                self.samples_width[index],
                self.samples_height[index],
            )
            self.plot_rectangle(
                rectangle,
                color=colors[index],
            )
