import unittest
import logging
import numpy as np
from functools import partial

from src.evolution import Recombination, Member
from src.target_function import ackley


class TestRecombinationMechanisms(unittest.TestCase):
    """
    Simple tests for member recombination
    """

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)
        self.optimum = Member(np.array([0]), ackley, [-30, 30], -1, Recombination.NONE)

    def test_none_recombination(self):
        child = self.optimum.recombine(Member(np.array([0]), ackley, [-30, 30], -1, Recombination.NONE))
        self.assertEqual(self.optimum.fitness, child.fitness)
        self.assertListEqual(self.optimum.x_coordinate.tolist(), child.x_coordinate.tolist())

    def test_uniform_recombination_2d_ackley(self):
        """
        Test that Recombination happens as expected
        """
        a = Member(np.array([0, 0]), ackley, [-30, 30], -1, Recombination.UNIFORM, recom_prob=.5)
        b = Member(np.array([1, 1]), ackley, [-30, 30], -1, Recombination.UNIFORM, recom_prob=.5)
        unique_entries, counts = np.unique([a.recombine(b).x_coordinate for _ in range(1_000)], axis=0,
                                           return_counts=True)
        self.assertEqual(np.sum(counts), 1_000)
        # we have exactly 4 possible combinations
        self.assertEqual(len(unique_entries), 4)
        # All are equally likely -> roughly 250 occurances per outcome
        self.assertTrue(np.allclose([250, 250, 250, 250], counts, rtol=10, atol=1))

    def test_uniform_recombination_2d_ackley_recom_prob_exception(self):
        """
        Test assertion error
        """
        a = Member(np.array([0, 0]), ackley, [-30, 30], -1, Recombination.UNIFORM)
        b = Member(np.array([1, 1]), ackley, [-30, 30], -1, Recombination.UNIFORM, recom_prob=.5)
        recombine_func = partial(a.recombine, b)
        self.assertRaises(AssertionError, recombine_func)

    def test_intermediate_recombination_1d_ackley(self):
        """Test child is actually in the middle"""
        a = Member(np.array([0]), ackley, [-30, 30], -1, Recombination.INTERMEDIATE)
        b = Member(np.array([0]), ackley, [-30, 30], -1, Recombination.INTERMEDIATE)
        offspring = a.recombine(b).x_coordinate
        mean = np.mean([a.x_coordinate, b.x_coordinate], axis=0)
        self.assertTrue(np.allclose(offspring, mean, rtol=.1, atol=.3))


