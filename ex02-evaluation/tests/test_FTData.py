import unittest
import logging

from src.main import Friedman_test

import numpy as np


class TestFTData(unittest.TestCase):
    """
    This test lets you confirm that if your Friedman Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        self.FTestData_data = np.loadtxt("./src/FTestData.csv", delimiter=',')

    def test_FTest(self):
        logging.info('Contents of Friedman_test:')
        chi2_F, FData_stats = Friedman_test(self.FTestData_data)
        self.assertAlmostEqual(chi2_F, 18.1333, places=3)

# Feel free to add more tests
