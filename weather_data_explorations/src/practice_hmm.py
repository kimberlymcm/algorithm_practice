import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from BaumWelchAlg import BaumWelch

hum_df = pd.read_csv("../data/historical-hourly-weather-data/humidity.csv",
                      index_col="datetime", parse_dates=["datetime"])
print("hum_df.size before filtering: ", hum_df.size)
hum_df = hum_df.between_time('11:30:00','12:30:00')
hum_df = hum_df[['Vancouver']]
hum_df = hum_df.rename(columns={"Vancouver": "van"})
print("hum_df.size after filtering for noon and NY: ", hum_df.size)

hum_df = hum_df.assign(van_impute=hum_df.van.fillna(hum_df.van.median()))


tfd = tfp.distributions

# Start with 2 states (high and low)
# And assume we know the probs
initial_distribution = tfd.Categorical(probs=[0.5, 0.5])

# Suppose a cold day has a 30% chance of being followed by a hot day
# and a hot day has a 20% chance of being followed by a cold day.
# We can model this as:
transition_distribution = tfd.Categorical(probs=[[0.95, 0.05],
                                                 [0.05, 0.95],
                                                ])

# Suppose additionally that on each day the temperature is
# normally distributed with mean and standard deviation 0 and 5 on
# a cold day and mean and standard deviation 15 and 10 on a hot day.
# We can model this with:

observation_distribution = tfd.Normal(loc=[95., 85.], scale=[5., 5.])

# Convert our observations into a tensor
observations = tf.convert_to_tensor(hum_df.van_impute)
observations = tf.cast(observations, tf.float32)
print("Observations")
print(observations)

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=1886)

fb = model.posterior_marginals(observations, mask=None, name=None)
fb.probs_parameter()


print(initial_distribution)
model =  BaumWelch(initial_distribution=initial_distribution,
                   observation_distribution=observation_distribution,
                   transition_distribution=transition_distribution,
                   num_steps=1886,
                   epsilon=0.0001,
                   maxStep=1886)

observations = tf.dtypes.cast(observations, dtype=tf.float32)
observations = tf.constant(observations, dtype=tf.float32, name='observation_sequence')
print(observations)

trans0, transition, emission, c = model.run_Baum_Welch_EM(observations, summary=False, monitor_state_1=True)