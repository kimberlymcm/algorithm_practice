""" This is used to create test Markov chains """

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils import generate_markov_chain


def main():
    # TODO (kmcmanus): Make these arguments

    tfd = tfp.distributions
    initial_distribution = tfd.Categorical(probs=[0.5, 0.5])
    transition_distribution = tfd.Categorical(probs=[[0.95, 0.05],
                                                     [0.05, 0.95],
                                                     ])
    observation_distribution = tfd.Normal(loc=[100., 20.], scale=[5., 5.])
    hidden_chain, observed_chain = generate_markov_chain(
        initial_distribution, transition_distribution, observation_distribution, 1886)

    with open('../data/hidden_chain.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % obs for obs in hidden_chain)
    with open('../data/observed_chain.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % obs for obs in observed_chain)


if __name__ == '__main__':
    main()
