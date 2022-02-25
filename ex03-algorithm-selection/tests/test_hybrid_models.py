import unittest
import logging

from src.aslib import select, get_stats


class TestSelectionHybridModels(unittest.TestCase):

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

        data = [['a', 1., 'A', 1., 'ok'],
                ['a', 1., 'B', 3., 'ok'],
                ['a', 1., 'C', 4., 'timeout'],

                ['b', 1., 'A', 2., 'ok'],
                ['b', 1., 'B', 1., 'ok'],
                ['b', 1., 'C', 4., 'timeout'],

                ['c', 1., 'A', 3., 'ok'],
                ['c', 1., 'B', 1., 'ok'],
                ['c', 1., 'C', 4., 'timeout'],

                ['d', 1., 'A', 1., 'ok'],
                ['d', 1., 'B', 3., 'ok'],
                ['d', 1., 'C', 4., 'timeout'],

                ['e', 1., 'A', 1., 'ok'],
                ['e', 1., 'B', 4., 'timeout'],
                ['e', 1., 'C', 0., 'ok'],

                ['f', 1., 'A', 5., 'timeout'],
                ['f', 1., 'B', 3., 'ok'],
                ['f', 1., 'C', 0., 'ok']]

        features = [['a', 0],
                    ['b', 1],
                    ['c', 1],
                    ['d', 0],
                    ['e', 2],
                    ['f', 2]]

        cv = [['a', 1, 1],
              ['b', 1, 1],
              ['c', 1, 2],
              ['d', 1, 2],
              ['e', 1, 3],
              ['f', 1, 3]]

        self.data = data
        self.features = features
        self.cv = cv

    def test_toy_data_simple(self):
        """
        With this simple toy data it should be easy to overfit such that we get oracle performance
        :return:
        """
        m, selection = select(self.data, self.features, self.cv, 4, 2, None, None, individual=False)
        o, s = get_stats(self.data, 4, 2)
        print(o, m, s)
        self.assertTrue(o <= m <= s)
        for feature, sel in zip(self.features, selection):  # selection should be perfectly matched to feature
            self.assertEqual(feature[1], sel)

# Feel free to add more tests
