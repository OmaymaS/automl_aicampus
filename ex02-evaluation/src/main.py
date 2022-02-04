import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def load_data_MNTest(fl="./MCTestData.csv"):
    """
    Loads data stored in McNemarTest.csv
    :param fl: filename of csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(fl, header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest(fl="./TMStTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest(fl="./FTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: evaluations
    """
    errors = np.loadtxt(fl, delimiter=",")
    return errors


def McNemar_test(labels, prediction_1, prediction_2):
    """
    TODO
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    chi2_Mc = np.random.uniform(0, 1)
    return chi2_Mc


def TwoMatchedSamplest_test(y1, y2):
    """
    TODO
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    t_value = np.random.uniform(0, 1)
    return t_value


def Friedman_test(errors):
    """
    TODO
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, you can add anything needed to facilitate
    solving the following post hoc problems
    """
    chi2_F = np.random.uniform(0, 1)
    FData_stats = {'errors': errors}
    return chi2_F, FData_stats


def Nemenyi_test(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    Q_value = np.empty_like([1])
    return Q_value


def box_plot(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    pass


def main(args):
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)

    # (d)
    Q_value = Nemenyi_test(FData_stats)

    # (e)
    box_plot(FData_stats)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)
