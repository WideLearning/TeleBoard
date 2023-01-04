import unittest
import numpy as np
from distributions import QuantileDistribution

def test_mean_for(population, q_size, s_size=10**6):
    qd = QuantileDistribution(population, q_size)
    estimates = [
        population.mean(),
        qd.sample(s_size).mean(),
        qd.mean(),
        qd.integrate(lambda x: x),
    ]
    print(estimates)
    return max(estimates) - min(estimates)

def test_var_for(population, q_size, s_size=10**6):
    qd = QuantileDistribution(population, q_size)
    estimates = [
        population.var(),
        qd.sample(s_size).var(),
        qd.integrate(lambda x: x**2) - qd.mean()**2,
    ]
    print(estimates)
    return max(estimates) - min(estimates)


class TestDistributions(unittest.TestCase):
    def test_mean(self):
        self.assertGreater(test_mean_for(np.random.standard_cauchy(1000), 1000), 0.1)
        self.assertLess(test_mean_for(np.random.standard_normal(1000), 1000), 0.1)
        self.assertLess(test_mean_for(np.random.standard_gamma(1, 1000), 1000), 0.1)
    
    def test_var(self):
        self.assertGreater(test_var_for(np.random.standard_cauchy(1000), 1000), 100)
        self.assertLess(test_var_for(np.random.standard_normal(1000), 1000), 0.1)
        self.assertLess(test_var_for(np.random.standard_gamma(1, 1000), 1000), 0.1)