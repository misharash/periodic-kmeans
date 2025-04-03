import numpy as np


def periodic_average_1d(a: np.ndarray[float], weights: np.ndarray[float] | None = None, period: float = 1):
    if a.ndim != 1: raise ValueError("a must be a one-dimensional ndarray")
    if weights is None: weights = np.ones_like(a)
    if weights.ndim != 1: raise ValueError("weights must be a one-dimensional ndarray")
    if weights.shape != a.shape: raise ValueError("weights must have the same shape as a")
    if (sum_w := weights.sum()) <= 0: raise ValueError("Sum of weights must be positive")
    if any(weight < 0 for weight in weights): raise ValueError("weights must not be negative")
    weights = weights / sum_w # normalize
    a = a % period # wrap "canonically" to [0, period)
    simple_average = np.average(a, weights = weights)
    period_2 = period / 2
    if a.max() - a.min() <= period_2: return simple_average # trivial case
    # if there exists a wrapping in which the range is narrower than period/2, it's enough to try [period/2, 3 period/2) in addition to the "canonical" one, as Miniak-Górecka, Podlaski and Gwizdałła 2022 suggest, here is a shorter implementation of the algorithm
    a2 = a + (a < period_2) * period
    if a2.max() - a2.min() <= period_2: return np.average(a2, weights = weights) % period
    # general case, when range is unavoidably wider than period/2
    # first, sort a and reorder weights in the same way
    idx = np.argsort(a)
    a = a[idx]
    weights = weights[idx]
    # in the following, we try to shift elements 0 through i by a period forward and see what happens to the weighted sum of squared differences of each element with the mean
    cumsum_w = np.cumsum(weights)
    new_averages = simple_average + cumsum_w * period # the new mean, the elements 0<=j<=i become (a[j]+period), simply increased by period, so it's only the sum of their weights that matters
    weighted_sums_of_squared_elements = np.sum(weights * a**2) + 2 * np.cumsum(weights * a) * period + cumsum_w * period**2 # the elements 0<=j<=i become (a[j] + period), so their squares increase by 2 a[j] period + period^2
    weighted_sums_of_squared_differences = weighted_sums_of_squared_elements - new_averages**2 # weighted sums of (a[j] - average)^2, expands into the weighted sum of a[j]^2 - average^2
    return new_averages[np.argmin(weighted_sums_of_squared_differences)] % period # the best average is the one that minimizes the weighted sum of squares