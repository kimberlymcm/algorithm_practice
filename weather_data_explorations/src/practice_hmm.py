""" Run Baum Welch on the humidity data """
from datetime import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from baum_welch_alg import BaumWelch


def read_data():
    hum_df = pd.read_csv("../data/historical-hourly-weather-data/humidity.csv",
                      index_col="datetime", parse_dates=["datetime"])
    print("hum_df.size before filtering: ", hum_df.size)
    hum_df = hum_df.between_time('11:30:00','12:30:00')
    hum_df = hum_df[['Vancouver']]
    hum_df = hum_df.rename(columns={"Vancouver": "van"})
    print("hum_df.size after filtering for noon and NY: ", hum_df.size)

    hum_df = hum_df.assign(van_impute=hum_df.van.fillna(hum_df.van.median()))
    observations = tf.cast(tf.convert_to_tensor(hum_df.van_impute), tf.float32)
    return observations


def main():

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "{}/run-{}/".format("../tf_logs", now)
    observations = read_data()
    observations = tf.constant(observations, dtype=tf.float32, name='observation_sequence')

    tfd = tfp.distributions
    # Note that 'scale' is the standard deviation, not the variance
    initial_distribution = tfd.Categorical(probs=[0.5, 0.5])
    transition_distribution = tfd.Categorical(probs=[[0.95, 0.05],
                                                     [0.05, 0.95],
                                                    ])
    observation_distribution = tfd.Normal(loc=[90., 70.], scale=[5., 5.])
    #initial_distribution = tfd.Categorical(probs=[0.33, 0.33, 0.34])
    #transition_distribution = tfd.Categorical(probs=[[0.95, 0.025, 0.025],
    #	                                             [0.025, 0.95, 0.025],
    #	                                             [0.025, 0.025, 0.95]])
    #observation_distribution = tfd.Normal(loc=[90., 70., 40.],
    #	                                  scale=[5., 2., 20.])
    model = BaumWelch(initial_distribution=initial_distribution,
                      observation_distribution=observation_distribution,
                      transition_distribution=transition_distribution,
                      num_steps=1886,
                      epsilon=0.02,
                      maxStep=50,
                      log_dir=logdir)
    initial_dist, trans_dist, observ_dist = model.run_Baum_Welch_EM(observations, summary=False, monitor_state_1=True)

if __name__ == '__main__':
	main()