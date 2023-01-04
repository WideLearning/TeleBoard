import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class QuantileDistribution:
    def __init__(self, points, k=100):
        assert points.ndim == 1
        points = points.astype(np.float32)
        self.k = k
        self.q = np.quantile(points, np.linspace(0, 1, k + 1), method="linear")

    def mean(self):
        return (self.q.sum() - (self.q[0] + self.q[-1]) / 2) / self.k

    def integrate(self, f):
        from scipy.integrate import quad

        segment = lambda l, r: quad(f, l, r)[0] / (self.k * max(r - l, 1e-18))
        return sum(segment(l, r) for l, r in zip(self.q, self.q[1:]))

    def sample(self, size):
        segments = np.random.randint(0, self.k, size)
        uniforms = np.random.uniform(0, 1, size)
        return self.q[segments] + uniforms * (self.q[segments + 1] - self.q[segments])