import unittest
import logging
import numpy as np

from src.evolution import ParentSelection, EA, Member
from src.target_function import ackley


class TestEvoAlgoPerf(unittest.TestCase):
    """
    Simple if performance improves over time
    """

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

    def test_parent_selection_tournament(self):
        """Test Tournament selection"""
        # With the fake population we should never choose member 1
        ea = EA(ackley, 2, 1, selection_type=ParentSelection.TOURNAMENT,
                total_number_of_function_evaluations=100, children_per_step=1)
        ea.population = [Member(np.array([0]), ackley, [-30, 30], -1, -1),
                         Member(np.array([30]), ackley, [-30, 30], -1, -1)]
        parent_ids = ea.select_parents()
        self.assertListEqual([0], parent_ids)

        # Test that we also rather choose 0 multiple times than to choose 1 even once
        ea = EA(ackley, 2, 1, selection_type=ParentSelection.TOURNAMENT,
                total_number_of_function_evaluations=100, children_per_step=2)
        ea.population = [Member(np.array([0]), ackley, [-30, 30], -1, -1),
                         Member(np.array([30]), ackley, [-30, 30], -1, -1)]
        parent_ids = np.array([ea.select_parents() for _ in range(100)]).flatten()
        self.assertTrue(1 not in parent_ids)

    def test_parent_selection_fitness(self):
        # With the fake population we should never choose member 1
        ea = EA(ackley, 5, 1, selection_type=ParentSelection.FITNESS,
                total_number_of_function_evaluations=100, children_per_step=1)
        ea.population = [Member(np.array([0]), ackley, [-30, 30], -1, -1),
                         Member(np.array([0.5]), ackley, [-30, 30], -1, -1),
                         Member(np.array([2]), ackley, [-30, 30], -1, -1),
                         Member(np.array([5]), ackley, [-30, 30], -1, -1),
                         Member(np.array([30]), ackley, [-30, 30], -1, -1)]

        uniques, counts = np.unique([ea.select_parents() for _ in range(1_000)], return_counts=True)
        print(uniques, counts)
        self.assertTrue(4 not in uniques)  # member 4 is so extremely bad it should never be sampled
        # the others are proportionally sampled according to their fitness
        self.assertTrue(counts[0] > counts[1] > counts[2] > counts[3])

    def test_parent_selection_neutral(self):
        ea = EA(ackley, 5, 1, selection_type=ParentSelection.NEUTRAL,
                total_number_of_function_evaluations=100, children_per_step=1)
        ea.population = [Member(np.array([0]), ackley, [-30, 30], -1, -1),
                         Member(np.array([0.5]), ackley, [-30, 30], -1, -1),
                         Member(np.array([2]), ackley, [-30, 30], -1, -1),
                         Member(np.array([5]), ackley, [-30, 30], -1, -1),
                         Member(np.array([30]), ackley, [-30, 30], -1, -1)]
        uniques, counts = np.unique([ea.select_parents() for _ in range(5_000)], return_counts=True)
        print(uniques, counts)
        self.assertEqual(5_000, np.sum(counts))
        self.assertListEqual([0, 1, 2, 3, 4], uniques.tolist())
        # zero-sum-game -> mean is easy to determine
        self.assertAlmostEqual(1_000, np.mean(counts))

