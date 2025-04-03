import numpy as np
from typing import Literal


def periodic_average_1d(a: np.ndarray[float], weights: np.ndarray[float] | None = None, period: float = 1):
    if a.ndim != 1: raise ValueError("a must be a one-dimensional ndarray")

    if weights is None: weights = np.ones_like(a) # equal weights by default, the number does not matter
    if weights.ndim != 1: raise ValueError("weights must be a one-dimensional ndarray")
    if weights.shape != a.shape: raise ValueError("weights must have the same length as a")
    if (sum_w := weights.sum()) <= 0: raise ValueError("Sum of weights must be positive")
    if any(weight < 0 for weight in weights): raise ValueError("weights must not be negative")

    weights = weights / sum_w # normalize
    a = a % period # wrap "canonically" to [0, period)
    simple_average = np.average(a, weights = weights)
    period_2 = period / 2
    if a.max() - a.min() <= period_2: return simple_average # trivial case
    # if there exists a wrapping in which the range is narrower than period/2, it's enough to try [period/2, 3 period/2) in addition to the "canonical" one, as Miniak-Górecka, Podlaski and Gwizdałła 2022 suggested, here is a shorter implementation of their algorithm
    a2 = a + (a < period_2) * period
    if a2.max() - a2.min() <= period_2: return np.average(a2, weights = weights) % period
    # general case, when range is unavoidably wider than period/2
    # first, sort a (coalescing the equal elements) and reorder (and collapse) weights in the same way
    a, idx = np.unique(a, return_inverse = True) # remove repeating elements (this involves sorting) but remember their original positions (idx is the same length as original a containing indices of unique elements in new a)
    weights = np.bincount(idx, weights = weights) # accumulate weights of the unique elements in the new a by adding old weight[j] for each idx[j] == i
    # in the following, we try to shift elements 0 through i by a period forward and see what happens to the weighted sum of squared differences of each element with the mean
    cumsum_w = np.cumsum(weights)
    new_averages = simple_average + cumsum_w * period # the new mean, the elements 0<=j<=i become (a[j]+period), simply increased by period, so it's only the sum of their weights that matters
    weighted_sums_of_squared_elements = np.sum(weights * a**2) + 2 * np.cumsum(weights * a) * period + cumsum_w * period**2 # the elements 0<=j<=i become (a[j] + period), so their squares increase by 2 a[j] period + period^2
    weighted_sums_of_squared_differences = weighted_sums_of_squared_elements - new_averages**2 # weighted sums of (a[j] - average)^2, expands into the weighted sum of a[j]^2 - average^2
    return new_averages[np.argmin(weighted_sums_of_squared_differences)] % period # the best average is the one that minimizes the weighted sum of squares


def periodic_average_2d(a: np.ndarray[float], axis: Literal[-2, -1, 0, 1] = 0, weights: np.ndarray[float] | None = None, period: float | np.ndarray[float] = 1):
    if a.ndim != 2: raise ValueError("a must be a two-dimensional ndarray")

    if axis > 1 or axis < -2: raise ValueError("Illegal axis for a two-dimensional ndarray")
    axis = (axis + 2) % 2 # turn negative axis to 0 or 1, so that the other axis is `not axis`
    if axis: a = a.T # if axis = 1 (or equivalently -1 originally), swap the axes

    if weights is None: weights = np.ones(a.shape[axis])
    if weights.ndim != 1: raise ValueError("weights must be a one-dimensional ndarray")
    if len(weights) != len(a): raise ValueError("weights must have the same length as a along the axis")
    if weights.sum() <= 0: raise ValueError("Sum of weights must be positive")
    if any(weight < 0 for weight in weights): raise ValueError("weights must not be negative")

    if np.shape(period) == tuple(): period = np.repeat(period, a.shape[not axis])
    if period.ndim != 1: raise ValueError("period must be a one-dimensional ndarray")
    if len(period) != a.shape[1]: raise ValueError("period must have the same length as a along the other axis")

    return np.array([periodic_average_1d(a[:, i], weights = weights, period = period[i]) for i in range(a.shape[1])])