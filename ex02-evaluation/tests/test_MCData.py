import unittest
import logging

from src.main import McNemar_test

import pandas as pd

class TestMCData(unittest.TestCase):
    """
    This test lets you confirm that if your McNemar Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        self.MCTest_data = pd.read_csv('./src/MCTestData.csv', header=None).to_numpy()

    def test_MCData(self):
        logging.info('Contents of McNemar_test:')
        chi2_Mc = McNemar_test(self.MCTest_data[:, 0], self.MCTest_data[:, 1], self.MCTest_data[:, 2])
        self.assertAlmostEqual(chi2_Mc, 4.614, places=3)

# Feel free to add more tests
