from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# TODO load a regressor from sklearn to solve exercise 2
# TODO load a classifier from sklearn to solve exercise 3

import numpy as np
import pandas as pd


def get_stats(aslib_data: List, cutoff: float, par: int = 10) -> (float, float):
    """
    Simple method to determine Virtual best and Single best performance.
    Expects input data in the ASLib data format.
    :param aslib_data: List of ASlib data.
                       Entries are as follows:
                       [('instance_id', 'STRING'),
                        ('repetition', 'NUMERIC'),
                        ('algorithm', 'STRING'),
                        ('runtime', 'NUMERIC'),
                        ('runstatus', ['ok', 'timeout', 'memout', 'not_applicable', 'crash', 'other'])]
    :param cutoff: The used cutoff as a float
    :param par: The penalization factor (default = 10) if runtime >= cutoff then runtime = PAR * cutoff
    :return: oracle_perf (float), single_best_perf(float)
    """
    df = pd.DataFrame(aslib_data)  # pandas data frames allow for easy data handling
    df.columns = ['instance_id', 'repetition', 'algorithm', 'runtime', 'runstatus']  # correctly name the entries
    algos = defaultdict(list)  # track individual algorithm performances
    insts = defaultdict(lambda: np.inf)  # track individual instance performances

    # SINGLE BEST
    df_grouped = df.groupby('algorithm')
    groups_keys = df_grouped.groups.keys()
    for k in groups_keys:
        df_partial = df.iloc[df_grouped.groups[k]]
        score_list = df_partial['runtime'].apply(lambda x: par*cutoff if (x >= cutoff) else x)
        algos[k]=score_list
    single_best = min([sum(score_list)/len(1+score_list) for score_list in algos.values()])

    ## VIRTUAL BEST 
    df_grouped = df.groupby('instance_id')
    groups_keys = df_grouped.groups.keys()
    for k in groups_keys:
        df_partial = df.iloc[df_grouped.groups[k]]
        score_list = df_partial['runtime'].apply(lambda x: par*cutoff if (x >= cutoff) else x)
        insts[k]=min(score_list)
    virtual_best=sum([score_list for score_list in insts.values()])/len(insts.values()) 
  
    return virtual_best,single_best

def hybrid_model(test_instances: List[str], algos: List[str], run_df: pd.DataFrame,
                 feature_df: pd.DataFrame, test_feature_df: pd.DataFrame) -> List[int]:
    """
    Use pairwise classification to predict which algorithm will outperform the others.
    Based on that, build your selection.
    :param test_instances: List of instance ids (str)
    :param algos: List of algorithms (str)
    :param run_df: Pandas Dataframe containing all training runtime data (i.e. y_train)
    :param feature_df: Pandas Dataframe containing all training feature data (i.e. X_train)
    :param test_feature_df: Pandas Dataframe containing all test feature data (i.e. X_test)
    :return: List of selected algorithms per test instance. I.e index of element in algos that should be used to solve
             an instance in test_instances
    """
    y_predictions = np.zeros((len(test_instances), len(algos)))
    X_train = feature_df.values[:, 1:]
    X_test = test_feature_df.values[:, 1:]

    # Use voting to decide which algorithm solves which instance
    min_idx = run_df.groupby('instance_id')['runtime'].idxmin()
    voted_algos=run_df.iloc[min_idx]
    y_train = voted_algos['algorithm']
    
    ## Fit model and predict
    log_reg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto').fit(X_train, y_train)
    y_predictions=log_reg.predict_proba(X_test) 
    selection = y_predictions.argmax(axis=1)
    
    return selection


def individual_model(test_instances: List[str], algos: List[str], run_df: pd.DataFrame,
                     feature_df: pd.DataFrame, test_feature_df: pd.DataFrame) -> List[int]:
    """
    Use any regression model you like to predict the performance for each algorithm individually.
    Based on this you should build your selection
    :param test_instances: List of instance ids (str)
    :param algos: List of algorithms (str)
    :param run_df: Pandas Dataframe containing all training runtime data (i.e. y_train)
    :param feature_df: Pandas Dataframe containing all training feature data (i.e. X_train)
    :param test_feature_df: Pandas Dataframe containing all test feature data (i.e. X_test)
    :return: List of selected algorithms per test instance. I.e index of element in algos that should be used to solve
             an instance in test_instances
    """
    y_predictions = np.zeros((len(test_instances), len(algos)))
    
    for idx, algo in enumerate(algos):
        y_train = run_df[run_df['algorithm']==algo]['runtime']
        X_train = feature_df.values[:, 1:]
        X_test = test_feature_df.values[:, 1:]
        reg = LinearRegression().fit(X_train, y_train)
        y_predictions[:,idx]=reg.predict(X_test)
        
    selection = y_predictions.argmin(axis=1)
    
    return selection


def select(aslib_run_data: List, aslib_feature_data: List, aslib_cv_splits: List,
           cutoff: int, parx: int, test_split_index: int = 10, algos: List = None,
           individual: bool = True
           ) -> (float, List[int]):
    """
    Method that trains individual regression models for each algorithm and predicts the performance on the final
    cv split.
    :param aslib_run_data: List of ASlib run data.
                           Entries are as follows:
                           [('instance_id', 'STRING'),
                            ('repetition', 'NUMERIC'),
                            ('algorithm', 'STRING'),
                            ('runtime', 'NUMERIC'),
                            ('runstatus', ['ok', 'timeout', 'memout', 'not_applicable', 'crash', 'other'])]
    :param aslib_feature_data: List of ASlib feature data.
                       Entries are as follows:
                       [('instance_id', 'STRING'),
                        ('feature_1', 'NUMERIC'),
                        ('feature_2', 'NUMERIC'),
                        ...,
                        ('feature_N', 'NUMERIC')]
    :param aslib_cv_splits: List of ASlib cv splits.
                       Entries are as follows:
                       [('instance_id', 'STRING'),
                        ('repetitions', 'NUMERIC'),
                        ('split_id', 'NUMERIC')]
    :param cutoff: int -> the maximum allowed runtime value
    :param parx: the penalization factor for timed-out runs.
    :param test_split_index: Which CV index is left out for evaluation purposes
    :param algos: List of algorithms to consider. If using the ASLib data is too expensive with all algorithms, you
                  can specify (as a list of strings) which algorithms you want to consider to speed up computation.
                  If set to None, all algorithms are considered
    :param individual: Boolean to determine if the individual method or the hybrid method should be used.
    :return: Mean performance on the split and corresponding selections
    """
    run_df = pd.DataFrame(aslib_run_data)  # pandas data frames allow for easy data handling
    run_df.columns = ['instance_id', 'repetition', 'algorithm', 'runtime', 'runstatus']  # correctly name the entries
    # replace timeouts with penalized runtime
    run_df['runtime'] = run_df['runtime'].apply(lambda runtime: runtime if runtime < cutoff else cutoff * parx)

    feature_df = pd.DataFrame(aslib_feature_data)
    cols = ['instance_id']
    for i in range(feature_df.shape[1] - 1):
        cols.append('feat_%d' % i)
    feature_df.columns = cols

    cv_df = pd.DataFrame(aslib_cv_splits)
    cv_df.columns = ['instance_id', 'repetition', 'split']

    if test_split_index:
        assert 0 < test_split_index < 11, 'Invalid split index. Only values from 1-10 are valid'
        # determine train and test instances
        train_instances = cv_df[cv_df['split'] != test_split_index]['instance_id'].values
        test_instances = cv_df[cv_df['split'] == test_split_index]['instance_id'].values
    else:  # train = test used for test purposes
        test_instances = train_instances = cv_df['instance_id'].unique()

    if not algos:
        algos = run_df['algorithm'].unique()  # all algorithms we need to fit models for
    else:
        run_df = run_df[run_df['algorithm'].isin(algos)]

    # we sort all entries according to the instances such that we have an easy match from run-data to feature-data
    test_run_df = run_df[run_df['instance_id'].isin(test_instances)].sort_values(['instance_id', 'algorithm'])
    run_df = run_df[run_df['instance_id'].isin(train_instances)].sort_values(['instance_id', 'algorithm'])
    test_feature_df = feature_df[feature_df['instance_id'].isin(test_instances)].sort_values('instance_id')
    feature_df = feature_df[feature_df['instance_id'].isin(train_instances)].sort_values('instance_id')

    # impute missing feature values (nan values) with mean values
    test_feature_df = test_feature_df.fillna(feature_df.mean())
    feature_df = feature_df.fillna(feature_df.mean())

    if individual:  # TODO complete the following methods
        selection = individual_model(test_instances, algos, run_df, feature_df, test_feature_df)
    else:
        selection = hybrid_model(test_instances, algos, run_df, feature_df, test_feature_df)

    performance = []
    for instance, sel in zip(test_instances, selection):
        row = test_run_df[(test_run_df['algorithm'] == algos[sel]) & (test_run_df['instance_id'] == instance)]
        performance.append(row['runtime'].iloc[0])
    mean = np.mean(performance)  # type: float
    # print(mean)
    return mean, selection
