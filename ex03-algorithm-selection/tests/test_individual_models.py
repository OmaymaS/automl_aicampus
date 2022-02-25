import unittest
import logging

from src.aslib import select, get_stats


class TestSelectionIndividualModels(unittest.TestCase):

    def setUp(self):  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

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

                ['f', 1., 'A', 5., 'timeout'],
                ['f', 1., 'B', 3., 'ok']]

        features = [['a', 0],
                    ['b', 1],
                    ['c', 1],
                    ['d', 0],
                    ['e', 0],
                    ['f', 1]]

        features_tricky = [['a', 0],
                           ['b', 1],
                           ['c', 1],
                           ['d', 0],
                           ['e', 1],
                           ['f', 0]]

        cv = [['a', 1, 1],
              ['b', 1, 1],
              ['c', 1, 2],
              ['d', 1, 2],
              ['e', 1, 3],
              ['f', 1, 3]]

        self.data = data
        self.features = features
        self.features_tricky = features_tricky
        self.cv = cv

    def test_toy_data_simple(self):
        """
        With this simple toy data it should be easy to overfit such that we get oracle performance
        :return:
        """
        m, selection = select(self.data, self.features, self.cv, 4, 2, None, None)
        o, s = get_stats(self.data, 4, 2)
        print(o, m, s)
        self.assertTrue(o <= m <= s)
        for feature, sel in zip(self.features, selection):  # selection should be perfectly matched to feature
            self.assertEqual(feature[1], sel)

    def test_toy_data_difficult(self):
        """
        More tricky but still manageable
        :return:
        """
        m, selection = select(self.data, self.features_tricky, self.cv, 4, 2, None, None)
        o, s = get_stats(self.data, 4, 2)
        print(o, m, s)
        self.assertTrue(o <= m <= s)
        # due to the change of instance features for instances of e and f we can't simply test the correct selection
        # like in the prior test

# Feel free to add more tests
