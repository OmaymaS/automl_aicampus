import unittest
import logging

from src.main import Friedman_test, Nemenyi_test

import numpy as np

class TestNemenyi(unittest.TestCase):
    """
    This test lets you confirm that if your implementation of Post-hoc Nemenyi Test correctly compute the test statistic
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        Q = np.array([[ 0., 0.92376043, 3.92598183, 1.84752086,  0.80829038],
                      [ 0., 0.,         3.0022214,  0.92376043,  -0.11547005],
                      [ 0., 0.,         0.,         -2.07846097, -3.11769145],
                      [ 0., 0.,         0.,         0.,          -1.03923048],
                      [ 0., 0.,         0.,         0.,          0.        ]])

        self.FTestData_data = np.loadtxt('./src/FTestData.csv', delimiter=',')

        self.Q = Q

    def test_FTest(self):
        _, FData_stats = Friedman_test(self.FTestData_data)

        logging.info('Content of Nemenyi_test')
        Q_value = Nemenyi_test(FData_stats)
        np.testing.assert_almost_equal(Q_value, self.Q, decimal=3)

# Feel free to add more tests

# Feel free to add more tests
