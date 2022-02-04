import unittest
import logging

from src.main import TwoMatchedSamplest_test

import numpy as np

class TestTMStData(unittest.TestCase):
    """
    This test lets you confirm that if your Two-Matched-Samples t-Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        self.TMStTest_data = np.loadtxt('./src/TMStTestData.csv', delimiter=',')

    def test_TMStTest(self):
        logging.info('Contents of TwoMatchedSamoles_test:')
        t_value = TwoMatchedSamplest_test(self.TMStTest_data[:, 0], self.TMStTest_data[:, 1])
        self.assertAlmostEqual(t_value, -8.9235, places=3)

# Feel free to add more tests

