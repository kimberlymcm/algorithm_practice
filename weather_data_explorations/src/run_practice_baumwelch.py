""" Run practice Baum Welch """

from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from baum_welch_alg import BaumWelch


def main():

    tfd = tfp.distributions
    initial_distribution = tfd.Categorical(probs=[0.5, 0.5])
    transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                     [0.5, 0.5],
                                                      ])
    observation_distribution = tfd.Normal(loc=[80., 10.], scale=[40., 2.])
    observations = [float(line.rstrip('\n'))
                    for line in open('../data/observed_chain.txt')]

    # Convert our observations into a tensor
    observations = tf.cast(tf.convert_to_tensor(observations), tf.float32)
    observations = tf.constant(
        observations,
        dtype=tf.float32,
        name='observation_sequence')

    now = datetime.utcnow().strftime("%Y%m%d%H%M")
    logdir = "tf_logs"
    logdir = "{}/run-{}/".format(logdir, now)

    model = BaumWelch(initial_distribution=initial_distribution,
                      observation_distribution=observation_distribution,
                      transition_distribution=transition_distribution,
                      num_steps=1886,
                      epsilon=0.05,
                      maxStep=1886,
                      log_dir=logdir)

    initial_dist, trans_dist, observ_dist = model.run_Baum_Welch_EM(
        observations, summary=False, monitor_state_1=True)


if __name__ == '__main__':
    main()
