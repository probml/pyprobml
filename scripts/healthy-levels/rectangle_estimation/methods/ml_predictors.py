from ..plot import MatplotlibMixin
from .. import calculate


class MLPredictor(MatplotlibMixin):

    def __init__(self):
        self.title_template = 'MLE predictive, n={}'
        self.filename_template = f'healthyLevels{self.__class__.__name__}{{}}uninfPrior.pdf'

    def fit(self, observations):
        self.title = self.title_template.format(len(observations))
        self.filename = self.filename_template.format(len(observations))
        self.smallest_rectangle_enclosing = calculate.smallest_rectangle_enclosing(observations)
        self.observations = observations

    def _plot(self):
        self.plot_points(self.observations)
        self.plot_rectangle(self.smallest_rectangle_enclosing, color='black')
