""" Utility functions for time series analysis """

import numpy as np

def generate_markov_chain(initial_distribution, transition_distribution,
	observation_distribution, num_steps):

    hidden_chain = []
    observed_chain = []
    init_state = np.random.choice([0, 1], p=initial_distribution.probs_parameter())
    curr_state = init_state

    for i in range(num_steps):
        new_state = np.random.choice([0, 1], p=np.array(transition_distribution.probs_parameter())[curr_state])
        hidden_chain.append(new_state)
        new_obs = observation_distribution.sample(1)[0][new_state]
        observed_chain.append(new_obs.numpy())
        curr_state = new_state

    return hidden_chain, observed_chain