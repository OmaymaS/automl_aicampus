import unittest
import logging

from src.aslib import get_stats


class TestStats(unittest.TestCase):
    """
    Simple tests that let you quickly check if you correctly handle the runtime data
    """

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

    def test_only_one_entry(self):
        data = [['a', 1., 'A', 1., 'ok']]
        o, s = get_stats(data, cutoff=100)
        self.assertEqual(o, s)
        self.assertEqual(o, 1.)

    def test_more_entries(self):
        data = [['a', 1., 'A', 1., 'ok'],
                ['a', 1., 'B', 3., 'ok'],

                ['b', 1., 'A', 2., 'ok'],
                ['b', 1., 'B', 1., 'ok'],

                ['c', 1., 'A', 3., 'ok'],
                ['c', 1., 'B', 1., 'ok'],

                ['d', 1., 'A', 1., 'ok'],
                ['d', 1., 'B', 3., 'ok'],

                ['e', 1., 'A', 1., 'ok'],
                ['e', 1., 'B', 4., 'timeout'],

                ['f', 1., 'A', 5., 'timeout'],  # even though a 5 is recorded as runtime value, the penalized value
                ['f', 1., 'B', 3., 'ok']]       # will be 4*2 instead of 5*2
        # O = 1+1+1+1+1+3   =  8/6 = 1.333333
        # A = 1+2+3+1+1+4*2 = 16/2 = 2.666666
        # B = 3+1+1+3+4*2+3 = 19/6 = 3.166666
        o, s = get_stats(data, cutoff=4, par=2)
        self.assertAlmostEqual(o, 1.333, places=3)
        self.assertAlmostEqual(s, 2.667, places=3)

# Feel free to add more tests