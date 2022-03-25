import unittest
import logging
import numpy as np

from src.evolution import Mutation, Member
from src.target_function import ackley


class TestMutationMechanisms(unittest.TestCase):
    """
    Simple tests for member mutation
    """

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)
        self.optimum = Member(np.array([0]), ackley, [-30, 30], Mutation.NONE, -1)

    def test_none_mutation(self):
        child = self.optimum.mutate()
        self.assertEqual(self.optimum.fitness, child.fitness)
        self.assertListEqual(self.optimum.x_coordinate.tolist(), child.x_coordinate.tolist())

    def test_uniform_mutation_2d_ackley(self):
        """
        Test that mutation happens frequently
        """
        member = Member(np.array([0, 0]), ackley, [-30, 30], Mutation.UNIFORM, -1)
        unique_entries, counts = np.unique([member.mutate().x_coordinate for _ in range(1_000)], axis=0,
                                           return_counts=True)
        self.assertEqual(np.sum(counts), 1_000)
        # we should have very few (if any) repetitions
        self.assertAlmostEqual(np.max(counts) / 1_000, 0, places=2)

    def test_uniform_mutation_border_2d_ackley(self):
        """Test we don't sample outside the borders"""
        member = Member(np.array([-1, -1]), ackley, [-1, 1], Mutation.UNIFORM, -1)
        offspring = np.vstack([member.mutate().x_coordinate for _ in range(1_000)])
        self.assertTrue(np.all((-1 <= offspring) & (offspring <= 1)))

    def test_gauss_mutation_2d_ackley_sigma_exception(self):
        """Test that hierarchical/conditional parameters have to be set for Gaussian"""
        self.assertRaises(AssertionError, Member(np.array([0, 0]), ackley, [-30, 30], Mutation.GAUSSIAN, -1).mutate)

    def test_gauss_mutation_2d_ackley(self):
        """Test we actually have 0 mean and stdev 1"""
        member = Member(np.array([0, 0]), ackley, [-30, 30], Mutation.GAUSSIAN, -1, sigma=1)
        offspring = np.vstack([member.mutate().x_coordinate for _ in range(1_000)])
        mean = np.mean(offspring, axis=0)
        stdev = np.std(offspring, axis=0)
        self.assertTrue(np.allclose([0., 0.], mean, rtol=.1, atol=.3))
        self.assertTrue(np.allclose([1., 1.], stdev, rtol=.1, atol=.3))

        # mean of the offspring should stay around the parent
        member = Member(np.array([10, -5]), ackley, [-30, 30], Mutation.GAUSSIAN, -1, sigma=6)
        offspring = np.vstack([member.mutate().x_coordinate for _ in range(1_000)])
        mean = np.mean(offspring, axis=0)
        stdev = np.std(offspring, axis=0)
        self.assertTrue(np.allclose([10., -5.], mean, rtol=.1, atol=.3))
        self.assertTrue(np.allclose([6., 6.], stdev, rtol=.1, atol=.3))  # sigma scales stdev

    def test_gauss_mutation_border_2d_ackley(self):
        """Test we can't sample outside borders"""
        member = Member(np.array([-1, -1]), ackley, [-1, 1], Mutation.GAUSSIAN, -1, sigma=1)
        offspring = np.vstack([member.mutate().x_coordinate for _ in range(1_000)])
        self.assertTrue(np.all((-1 <= offspring) & (offspring <= 1)))


