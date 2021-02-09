import numpy as np

def generate_space(from_, to_, step):
    x = np.arange(from_, to_+step, step)
    xs, ys = np.meshgrid(x, x)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    return np.column_stack([xs, ys])


def smallest_rectangle_enclosing(observations):
    x_min = np.min(observations[:,0], 0)
    x_max = np.max(observations[:,0], 0)
    y_min = np.min(observations[:,1], 0)
    y_max = np.max(observations[:,1], 0)
    x_span = x_max - x_min
    y_span = y_max - y_min
    return x_min, y_min, x_span, y_span
