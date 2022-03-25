import numpy as np


def ackley(coordinate: np.ndarray) -> float:
    """
    n-dimensional Ackley function. Bounded by -30 <= coordinate[i] <= 30
    :param coordinate: n-dimensional numpy array with dtype float
    :return: function value at the given coordinate
    """
    assert np.all((-30 <= coordinate) & (coordinate <= 30)), 'Coordinates have to be in [-30, 30]'
    first_sum = 0.0
    second_sum = 0.0
    for c in coordinate:
        first_sum += c ** 2.0
        second_sum += np.cos(2.0 * np.pi * c)
    n = float(len(coordinate))
    return -20.0 * np.exp(-0.2 * np.sqrt(first_sum / n)) - np.exp(second_sum / n) + 20 + np.e
