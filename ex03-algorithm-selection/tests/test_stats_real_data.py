import unittest
import logging

import os
import arff  # needed to load ASLib data

from src.aslib import get_stats


class TestStatsRealData(unittest.TestCase):
    """
    This test lets you further confirm that you correctly compute the single best and oracle performances
    """

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        # get data dir relative to this directory
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
        indu_fh = os.path.join(data_dir, 'SAT11-INDU')
        rand_fh = os.path.join(data_dir, 'SAT11-RAND')
        self.indu_algo_runs = arff.load(open(os.path.join(indu_fh, 'algorithm_runs.arff'), 'r'))
        # self.indu_algo_feats = arff.load(open(os.path.join(indu_fh, 'feature_values.arff'), 'r'))
        self.rand_algo_runs = arff.load(open(os.path.join(rand_fh, 'algorithm_runs.arff'), 'r'))
        # self.rand_algo_feats = arff.load(open(os.path.join(rand_fh, 'feature_values.arff'), 'r'))
        self.cutoff = 5_000  # in python you can use underscore to make large numbers easier readable
        self.parx = 10

    def test_indu(self):
        logging.info('Contents of indu_algo_runs["data"]:')
        # attributes contains detailed descriptions of all entries of the data
        # logging.info(self.indu_algo_runs['attributes'])

        oracle_perf, single_best_perf = get_stats(self.indu_algo_runs['data'], cutoff=self.cutoff, par=self.parx)
        self.assertAlmostEqual(oracle_perf, 8187.5, 1)
        self.assertAlmostEqual(single_best_perf, 14605.9, 1)

    def test_rand(self):
        logging.info('Contents of rand_algo_runs["data"]:')
        # attributes contains detailed descriptions of all entries of the data
        logging.info(self.rand_algo_runs['attributes'])

        oracle_perf, single_best_perf = get_stats(self.rand_algo_runs['data'], cutoff=self.cutoff, par=self.parx)
        self.assertAlmostEqual(oracle_perf, 9186.4, 1)
        self.assertAlmostEqual(single_best_perf, 19916.4, 1)

# Feel free to add more tests