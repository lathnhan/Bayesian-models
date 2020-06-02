from machine_learning.ml_utils.feature_extractor.feature_matrix import FeatureExtractor
from machine_learning.ml_utils.ts_cross_validation.tsc import ExpandingCrossValidator  # I changed this
from machine_learning.ml_utils.returns.returns import calc_future_returns
from machine_learning.ml_utils.predict_collectors import PredictionsCollector
from machine_learning.ml_utils import helpers

import pystan

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np

# In[2]:
import multiprocessing as mp

import matplotlib.pyplot as plt
import pickle
import pyarrow


def run_bayesian_model(idx,
                       train_interval,
                       val_interval,
                       config):
    """
    This is the main function to run the Bayesian Model on Spark Cluster.
    Can be created using Jupyter Notebook.
    Args:
        idx: The index of training/Validation period
        train_interval: Training period indices
        val_interval: Prediction period indices
        config: Rolling Configuration file

    Returns: None

    """

    start_date = config['start_data_date']
    end_date = config['end_data_date']
    print(idx)
    start_t, end_t = helpers.get_ts_range(train_interval)
    print('idx {}: Train range: Start={}/End={}'.format(idx, start_t, end_t))
    start_v, end_v = helpers.get_ts_range(val_interval)
    print('idx {}: Validate range: Start={}/End={}'.format(idx, start_v, end_v))

    output_dir = '/media/farmshare2/Research/anthonyk/spark_bayesian/FCF_SI_event_beta_mcap1/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the Output Directory for each Training Period
    output_path_period = output_dir + '/training-period_{}'.format(idx)
    if not os.path.exists(output_path_period):
        os.makedirs(output_path_period)

    output_path_plots = output_path_period + '/plots'
    if not os.path.isdir(output_path_plots):
        os.mkdir(output_path_plots)

    PATH_FEATURES_DATA = '/media/farmshare2/Research/FEATURE_ENGINEERING/SI_FCF/ranked'

    PATH_POOL_DATA = '/media/farmshare2/Research/anthonyk/spark_bayesian/mcap'
    #PATH_POOL_DATA = '/media/farmshare2/_PoolDump/RUSSELL3000_CONSOLIDATED/DATA-625/20190415/2000_2015/LATEST'

    PATH_RETURNS_DATA = '/media/farmshare2/_PoolDump/RUSSELL3000_CONSOLIDATED/DATA-625/20190415/2000_2015/LATEST/FWDReturns'

    POOL_INDICATOR = '{}/mcap1.csv'.format(PATH_POOL_DATA)
    #POOL_INDICATOR = '{}/_poolIndicatorF.csv'.format(PATH_POOL_DATA)

    feature_map = {
        'SI' : '{}/RS3K_SI_TRUNC20_PR_FF.csv'.format(PATH_FEATURES_DATA),
        'FCF' : '{}/RS3K_FCFoMCap_TRUNC80_PR_EVENT.csv'.format(PATH_FEATURES_DATA),
        'Beta' : '/media/farmshare2/_PoolDump/RUSSELL3000_CONSOLIDATED/DATA-625/20190415/2000_2015/LATEST/_BR_BetaVsRUSSELL3000_Fast_ReleasedF.csv'
    }

    target_map = {
        'FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60': '{}/FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60.csv'.format(PATH_RETURNS_DATA)
    }

    # Preparing the feature and target data

    # Read in basic data
    pool_indicator = pd.read_csv(POOL_INDICATOR, index_col=0, parse_dates=True)

    # load all the feature data
    features = helpers.load_data_map(feature_map, pool_indicator, start_date, end_date)

    # Load the target return data
    target_dict = helpers.load_data_map(target_map, pool_indicator, start_date, end_date)

    # Shift the target returns by -1
    target_dict['FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60'] = target_dict['FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60'].shift(-1)

    horizon_days = 5
    is_open_next_day = True
    train_test_gap = horizon_days + (1 if is_open_next_day else 0)

    feature_extractor = FeatureExtractor(features,
                                         target=target_dict['FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60'],
                                         target_name='FWD_Alpha-MCWTR-BrBetaR3KFast_O2O_W60')

    # Defining the Stan Model

    model_name = 'Complex_Model'
    sampling_rate = 1.0  # 0.15
    #num_samples = 9e4
    group_indicator = 'None'
    num_iterations = 500  # 500
    num_chains = 5  # 7

    model_code_complex = """
 data {
        int<lower=0> N;
        vector[N] SI;
        vector[N] FCF;
        vector[N] Beta;
        vector[N] Y;
    }

    parameters {
        real intercept;
        real si_coef;
        real fcf_coef;  
        real beta_coef;
        real<lower=0,upper=100> sigma;
        real<lower=1,upper=20> nu;
    }

    model {
        intercept ~ normal(0, 1.0);
        si_coef ~ normal(0, 1.0);
        fcf_coef ~ normal(0, 0.5);
        beta_coef ~ normal(0,0.5);
        sigma ~ uniform(0, 100);
        nu ~ uniform(1, 20);
        Y ~ student_t(nu, intercept + si_coef * SI + fcf_coef * FCF + beta_coef * Beta , sigma);
    }
    """


    # Preparing the Data for STAN Model

    print('Extracting training data...')
    X_train, y_train, index_train = (
        feature_extractor.get_slice(datetime_range=train_interval))

    print('Extracting testing data...')
    X_val, y_val, index_val = (
        feature_extractor.get_slice(datetime_range=val_interval))

    feature_names = list(feature_extractor.feature_names)

    # Sampling the training set
    sampling_index = range(X_train.shape[0])
    np.random.seed(2019)
    #sampling_index = np.random.choice(sampling_index, replace=False, size=int(num_samples))
    sampling_index = np.random.choice(sampling_index, replace=False, size=int(len(sampling_index)*sampling_rate))

    X_train = X_train[sampling_index, :]
    y_train = y_train[sampling_index]
    index_train = index_train.iloc[sampling_index]

    model_data_dict = {}
    model_data_dict['N'] = X_train.shape[0]
    model_data_dict['Y'] = y_train
    for feature_name in feature_names:
        model_data_dict[feature_name] = X_train[:, feature_names.index(feature_name)]

    # Fitting the stan model
    stan_model = pystan.StanModel(model_code=model_code_complex, verbose=True)
    model_fit = stan_model.sampling(data=model_data_dict, iter=num_iterations, chains=num_chains, n_jobs=num_chains,
                                    seed=2019)

    parameter_samples = model_fit.extract(permuted=True)

    # Saving The Output of Model

    for parameter_name in parameter_samples:
        plt.figure()
        model_fit.plot(pars=[parameter_name])
        plt.savefig(output_path_plots + '/chain_{}_{}.pdf'.format(parameter_name, idx))
        plt.close()

    # Getting the scores from the model
    X_val_df = pd.DataFrame(X_val, columns=feature_names)

    #This was the part that had a bug in the original bayesian_ML2 file: Fixed
    scores = X_val_df.apply(lambda x: pd.Series(parameter_samples['intercept'] + x['SI'] * parameter_samples['si_coef']
                                                + x['FCF'] * parameter_samples['fcf_coef'] + x['Beta'] * parameter_samples['beta_coef']), axis=1)

    scores = pd.concat([index_val, scores], axis=1)

    # Need this so it can save as a parquet
    scores.columns = ['t()', 'Ticker'] + list(map(str, scores.columns[2:]))
    scores.set_index(['t()']).to_parquet(output_path_period + '/posterior-samples_{}.parquet'.format(idx))

    with open(output_path_period + '/parameter-realisations_{}.pkl'.format(idx), 'wb') as handle:
        pickle.dump(parameter_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path_period + '/stan-model_{}.pkl'.format(idx), 'wb') as f:
        pickle.dump(model_fit, f)
