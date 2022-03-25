import unittest
import logging
import numpy as np

from src.hpo import determine_best_hypers
from src.evolution import Recombination


class TestSimpleHPO(unittest.TestCase):

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

    def test_hpo(self):
        """Test resulting best configuration when overfitting to the ackley function"""
        # We simply test if we always come close to the optimum
        all_configs = []
        all_perfs = []
        for i in range(50):  # we evaluate over multiple seeds to get a better estimate of the true performance
            config, perf = determine_best_hypers()
            # Configs have to be tuples in the following order (Mutation, Selection, Recombination)
            all_configs.append(np.array(config))
            all_perfs.append(perf)
        self.assertAlmostEqual(np.mean(all_perfs), 0., places=1)

        # By nature of the ackley function the Intermediate recombination strategy is expected to be much more
        # preferable than any other recombination strategy
        uniques, counts = np.unique(np.array(all_configs)[:, 2], return_counts=True, axis=0)
        counts = counts.tolist()
        uniques = uniques.tolist()
        self.assertTrue(int(Recombination.INTERMEDIATE) in uniques)
        inter_count = counts.pop(uniques.index(int(Recombination.INTERMEDIATE)))
        # so we can simply test if it was found to be best more than all other recombination methods combined
        self.assertTrue(inter_count > np.sum(counts))



if __name__ == '__main__':
    unittest.main()
