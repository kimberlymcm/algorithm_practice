# Partially from https://github.com/geeky-bit/Tensorflow-HiddenMarkovModel-Baum_Welch-Viterbi-forward_backward-algo/blob/master/HiddenMarkovModel.py
import sys


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.seed_stream import SeedStream


def generate_chain(initial_distribution, transition_distribution, observation_distribution,
    num_steps):
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


class BaumWelch(tfp.distributions.HiddenMarkovModel):
    # When you add an __init__ to the child class, it will no longer
    # inherit the init from the parent class
    # The child init overrides
    # To keep the parents init, have to do: Person.__init__(self, fname, lname)

    """
   This is the Hidden Markov Model Class
    -----------
    
    - S: Number of states.
    - T: Transition matrix of size S by S  stores probability from state i to state j
    - E: Emission matrix of size S by N (number of observations) stores the probability of observing  O_j  from state  S_i 
    - T0: Initial state probabilities of size S.
    """

    def __init__(self, initial_distribution, observation_distribution, transition_distribution, num_steps, epsilon = 0.05, maxStep = 10):

        super().__init__(initial_distribution, transition_distribution, observation_distribution, num_steps)
  
        with tf.name_scope('Inital_Parameters'):
            with tf.name_scope('Scalar_constants'):
                
                # Max number of iteration
                self.maxStep = maxStep

                # convergence criteria  
                self.epsilon = epsilon 

                # Number of possible states
                self.S = np.int(self.initial_distribution.num_categories)

                # Number of possible observations
                self.O = np.int(self.initial_distribution.num_categories)
                
                self.prob_state_1 = []

                self.forward_log_probs = tf.Variable(tf.zeros(self.O, dtype=tf.float64), name='forward')
                self.backward_log_probs = tf.Variable(tf.zeros(self.O, dtype=tf.float64), name='backward')
                self.fb_array = tf.Variable(tf.zeros(self.O, dtype=tf.float64), name='posterior')


    def forward_backward(self, obs_prob_seq, x):
        """
        runs forward backward algorithm on observation sequence
        Arguments
        ---------
        - obs_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states
        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """
        obs_prob_list_for = tf.split(obs_prob_seq, self.N, 0) # creates a list of tensors for each observation
        # Below is identical to obs_prob_seq, except it is logs
        observation_log_probs = self._observation_log_probs(x, mask=None)     
        with tf.name_scope('forward_belief_propagation'):
            # forward belief propagation
            self._forward(observation_log_probs)
        with tf.name_scope('backward_belief_propagation'):
            # backward belief propagation
            self._backward(observation_log_probs)


    def _forward(self, observation_log_probs):
    
        with tf.name_scope('forward_first_step'):
            # prob_starting_in_each_state * prob_of_observed_state_given_hiddenstate[0]
            init_prob = self.initial_distribution.logits_parameter() + tf.squeeze(observation_log_probs[0])
            log_transition = self.transition_distribution.logits_parameter()
            log_adjoint_prob = tf.zeros_like(init_prob)

            
            def _scan_multiple_steps_forwards():
                def forward_step(log_previous_step, log_prob_observation):
                    return _log_vector_matrix(log_previous_step,
                                              log_transition) + log_prob_observation

                forward_log_probs = tf.scan(forward_step, observation_log_probs[1:],
                                    initializer=init_prob,
                                    name="forward_log_probs")
                return tf.concat([[init_prob], forward_log_probs], axis=0)

            forward_log_probs = prefer_static.cond(
                self._num_steps > 1,
                _scan_multiple_steps_forwards,
                lambda: tf.convert_to_tensor([init_prob]))

            total_log_prob = tf.reduce_logsumexp(forward_log_probs[-1], axis=-1)

        self.forward_log_probs = forward_log_probs
    

    def _backward(self, observation_log_probs):
        init_prob = self.initial_distribution.logits_parameter() + tf.squeeze(observation_log_probs[0])
        log_transition = self.transition_distribution.logits_parameter()
        log_adjoint_prob = tf.zeros_like(init_prob)

        def _scan_multiple_steps_backwards():
            """Perform `scan` operation when `num_steps` > 1."""

            def backward_step(log_previous_step, log_prob_observation):
              return _log_matrix_vector(
                  log_transition,
                  log_prob_observation + log_previous_step)

            backward_log_adjoint_probs = tf.scan(
                backward_step,
                observation_log_probs[1:],
                initializer=log_adjoint_prob,
                reverse=True,
                name="backward_log_adjoint_probs")

            return tf.concat([backward_log_adjoint_probs,
                              [log_adjoint_prob]], axis=0)

        backward_log_adjoint_probs = prefer_static.cond(
            self._num_steps > 1,
            _scan_multiple_steps_backwards,
            lambda: tf.convert_to_tensor([log_adjoint_prob]))
        self.backward_log_probs = backward_log_adjoint_probs

        
    def _posterior(self):
        # posterior score
        self.posterior = tf.multiply(self.forward, self.backward)

        marginal = tf.reduce_sum(self.posterior, 1)
        self.posterior = self.posterior / tf.expand_dims(marginal, 1)   

        
    def re_estimate_emission(self, x):
        
        # pg 73: uj(t)*x(t)
        tmp = tf.multiply(tf.math.exp(self.fb_array), tf.expand_dims(x, 1))
        emission_score = tf.math.reduce_sum(tmp, 0)
        emission_score_log = tf.math.log(emission_score)

        denom = tf.math.reduce_logsumexp(self.fb_array, 0)
        means_log = emission_score_log - denom
        means = tf.math.exp(means_log)

        new_stds = []
        for i in range(self.O):
            tmp_0 = (x - tf.math.exp(means_log[i]))**2
            tmp_1 = tf.multiply(tf.math.exp(self.fb_array[:, i]), tmp_0) # not logs
            variance_score = tf.math.reduce_sum(tmp_1, 0)
            new_var = variance_score / tf.math.exp(denom[i])
            new_std = tf.math.sqrt(new_var)
            new_stds.append(new_std)

        new_emissions = tfp.distributions.Normal(loc=means, scale=new_std)

        return new_emissions


    def re_estimate_transition(self, x, fb_array):
        
        with tf.name_scope('Init_3D_tensor'):
            # What is M??
            # M: v tensor (pg 70): num_states x num_states x num_observations
            # u (pg 70) is the fb_array I think
            self.M = tf.Variable(tf.zeros((self.N-1, self.S, self.S), tf.float32))
            # I think below should be the real denominator
            total_log_prob = tf.reduce_logsumexp(self.forward_log_probs[-1], axis=-1)
        
        with tf.name_scope('3D_tensor_transition'):
            # For each observation
            for t in range(self.N - 1):
                with tf.name_scope('time_step-%s' %t):
                    # tmp_0 = prob of being in state * prob of transitioning out of state
                    # eq 4.14 pg 71
                    tmp_00 = tf.expand_dims(self.forward_log_probs[t, :], 0)
                    tmp_0 = _log_vector_matrix(tmp_00, self.transition_distribution.logits_parameter())

                    tmp_1 = tmp_0 + self._observation_distribution.log_prob(x[t+1]) 

                    denom = tf.squeeze(_log_vector_matrix(tmp_1, self.backward_log_probs[t+1, :]))

                with tf.name_scope('Init_new_transition'):
                    trans_re_estimate = tf.Variable(tf.zeros((self.S, self.S), tf.float32))
                
                # For each state:
                #     forward[t, i] * transition[1->0, 1->1] * observation[1|0, 1|0] * backward[t+1]
                for i in range(self.S):
                    with tf.name_scope('State-%s' %i):
                        # KM tried to fix
                        # P(being in state i at time t) * [P(trans i->j), P(i->i)]
                        tmp_0 = self.forward_log_probs[t, i] + np.array(self.transition_distribution.logits_parameter())[i, :]
                        # [P(state i time t & [P(i->i), P(i->i)]) * [P(o|i), P[o|j]]
                        # [P(state i time t & P(i->i) & P(o|i) & P(state i time t+1))]
                        # [P(state i time t & P(i->j) & P(o|j) & P(state ij time t+1))]
                        numer = tmp_0 + self._observation_distribution.log_prob(x[t+1]) + self.backward_log_probs[t+1, :]
                        trans_re_estimate = tf.compat.v1.scatter_update(trans_re_estimate, i, numer - total_log_prob)

                self.M = tf.compat.v1.scatter_update(self.M, t, trans_re_estimate)


        with tf.name_scope('Smooth_gamma'):

            numer = tf.reduce_logsumexp(self.M, 0) #4x4, need to do this to get sums of logs
            denom = tf.reduce_logsumexp(numer, 0) # Might need to be over axis 1, not 0
            T_new_tmp = numer - denom
            T_new = tfp.distributions.Categorical(logits=T_new_tmp)
        
        with tf.name_scope('New_init_states_prob'):
            # #1 pg 72
            T0_new = tfp.distributions.Categorical(logits=self.fb_array[0, :])
        
        return T0_new, T_new

  
    def check_convergence(self, new_T0, new_transition, new_emission):
        delta_T0 = tf.reduce_max(
                       tf.abs(
                           self.initial_distribution.probs_parameter() - new_T0.probs_parameter()
                        ) / self.initial_distribution.probs_parameter()
                       ) < self.epsilon
        print(tf.reduce_max(
                       tf.abs(
                           self.initial_distribution.probs_parameter() - new_T0.probs_parameter()
                        ) / self.initial_distribution.probs_parameter()
                       ))
        delta_T = tf.reduce_max(
                        tf.abs(
                            self.transition_distribution.probs_parameter() - new_transition.probs_parameter()
                       ) / self.transition_distribution.probs_parameter()
                    ) < self.epsilon
        print(tf.reduce_max(
                        tf.abs(
                            self.transition_distribution.probs_parameter() - new_transition.probs_parameter()
                       ) / self.transition_distribution.probs_parameter()
                    ))
        delta_E = tf.reduce_max(
                        tf.abs(
                            self.observation_distribution.mean() - new_emission.mean()
                        ) / self.observation_distribution.mean()
                    ) < self.epsilon
        print(tf.reduce_max(
                        tf.abs(
                            self.observation_distribution.mean() - new_emission.mean()
                        ) / self.observation_distribution.mean()
                    ))

        return tf.logical_and(tf.logical_and(delta_T0, delta_T), delta_E)
        

    def expectation_maximization_step(self, x): # x is the observations

        obs_prob_seq = tf.math.exp(self._observation_log_probs(x, mask=None)) # NOT LOGS

        ## probability of emission sequence
        with tf.name_scope('Forward_Backward'):
            self.forward_backward(obs_prob_seq, x)

            total_log_prob = tf.reduce_logsumexp(self.forward_log_probs[-1], axis=-1)
            log_likelihoods = self.forward_log_probs + self.backward_log_probs
            marginal_log_probs = distribution_util.move_dimension(
                log_likelihoods - total_log_prob[..., tf.newaxis], 0, -2)
            self.fb_array = marginal_log_probs


        with tf.name_scope('Re_estimate_transition'):
            new_T0, new_transition = self.re_estimate_transition(x, self.fb_array)
        
        with tf.name_scope('Re_estimate_emission'):
            new_emission = self.re_estimate_emission(x)

        with tf.name_scope('Check_Convergence'):
            print("NEW T0")
            print(new_T0.probs_parameter())
            print("NEW TRANSITION")
            print(new_transition.probs_parameter())
            print("NEW EMISSION")
            print(new_emission.mean())
            print(new_emission.variance())
            converged = self.check_convergence(new_T0, new_transition, new_emission)

        with tf.name_scope('Update_parameters'):
            #self.initial_distribution = tf.compat.v1.assign(self.initial_distribution, new_T0)
            #self.observation_distribution = tf.compat.v1.assign(self.observation_distribution, new_emission)
            #self.transition_distribution = tf.compat.v1.assign(self.transition_distribution, new_transition)
            self._initial_distribution = new_T0
            self._observation_distribution = new_emission
            self._transition_distribution = new_transition # WHAT IS THE POINT OF TF.ASSIGN

        return converged
        
 
    def Baum_Welch_EM(self, obs_seq):
        
        converged = tf.cast(False, tf.bool)
        
        with tf.name_scope('Train_Baum_Welch'):
            for i in range(self.maxStep): # Loop through each step
                
                with tf.name_scope('EM_step-%s' %i):
                    converged = self.expectation_maximization_step(obs_seq)
                    if converged == True:
                        print("CONVERGED")
                        break

        return converged

  
    def run_Baum_Welch_EM(self, observations, summary=False, monitor_state_1=False):


        with tf.name_scope('Input_Observed_Sequence'):
            self.N = len(np.array(observations))  # length of observed sequence
        
        # self.forward_log_probs, self.backward_log_probs, self.fbarray        
        fb_array = self.posterior_marginals(observations, mask=None, name=None)
        
        # Now runs Baum Welch
        converged = self.Baum_Welch_EM(observations)
        
        #with tf.Session() as sess:
            
        #    sess.run(tf.global_variables_initializer())
        #    trans0, transition, emission, c = sess.run([self.T0, self.T, self.E, converged])
            
            #if monitor_state_1:
            #    self.state_summary = np.array([sess.run(g) for g in self.prob_state_1])
            
            #if summary:
                # Instantiate a FileWriter to output summaries and the Graph.
            #    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
            #    summary_str = sess.run(summary_op)
            #    summary_writer.add_summary(summary_str)

        #    return trans0, transition, emission, c


    def posterior_marginals(self, observations, mask=None, name=None):
        """Compute marginal posterior distribution for each state.
        This function computes, for each time step, the marginal
        conditional probability that the hidden Markov model was in
        each possible state given the observations that were made
        at each time step.
        So if the hidden states are `z[0],...,z[num_steps - 1]` and
        the observations are `x[0], ..., x[num_steps - 1]`, then
        this function computes `P(z[i] | x[0], ..., x[num_steps - 1])`
        for all `i` from `0` to `num_steps - 1`.
        This operation is sometimes called smoothing. It uses a form
        of the forward-backward algorithm.
        Note: the behavior of this function is undefined if the
        `observations` argument represents impossible observations
        from the model.
        Args:
          observations: A tensor representing a batch of observations
            made on the hidden Markov model.  The rightmost dimension of this tensor
            gives the steps in a sequence of observations from a single sample from
            the hidden Markov model. The size of this dimension should match the
            `num_steps` parameter of the hidden Markov model object. The other
            dimensions are the dimensions of the batch and these are broadcast with
            the hidden Markov model's parameters.
          mask: optional bool-type `tensor` with rightmost dimension matching
            `num_steps` indicating which observations the result of this
            function should be conditioned on. When the mask has value
            `True` the corresponding observations aren't used.
            if `mask` is `None` then all of the observations are used.
            the `mask` dimensions left of the last are broadcast with the
            hmm batch as well as with the observations.
          name: Python `str` name prefixed to Ops created by this class.
            Default value: "HiddenMarkovModel".
        Returns:
          posterior_marginal: A `Categorical` distribution object representing the
            marginal probability of the hidden Markov model being in each state at
            each step. The rightmost dimension of the `Categorical` distributions
            batch will equal the `num_steps` parameter providing one marginal
            distribution for each step. The other dimensions are the dimensions
            corresponding to the batch of observations.
        Raises:
          ValueError: if rightmost dimension of `observations` does not
          have size `num_steps`.
        """

        with tf.name_scope(name or "posterior_marginals"):
          with tf.control_dependencies(self._runtime_assertions):
            observation_tensor_shape = tf.shape(observations)
            mask_tensor_shape = tf.shape(mask) if mask is not None else None

            with self._observation_mask_shape_preconditions(
                observation_tensor_shape, mask_tensor_shape):
              observation_log_probs = self._observation_log_probs(
                  observations, mask)
              log_prob = self._log_init + observation_log_probs[0]
              #log_transition = self._log_trans
              log_transition = self.transition_distribution.logits_parameter()
              log_adjoint_prob = tf.zeros_like(log_prob)

              def _scan_multiple_steps_forwards():
                def forward_step(log_previous_step, log_prob_observation):
                  return _log_vector_matrix(log_previous_step,
                                            log_transition) + log_prob_observation
 
                forward_log_probs = tf.scan(forward_step, observation_log_probs[1:],
                                            initializer=log_prob,
                                            name="forward_log_probs")
                return tf.concat([[log_prob], forward_log_probs], axis=0)

              forward_log_probs = prefer_static.cond(
                  self._num_steps > 1,
                  _scan_multiple_steps_forwards,
                  lambda: tf.convert_to_tensor([log_prob]))

              # KM added to try to make mutable
              forward_log_probs = tf.Variable(forward_log_probs)

              total_log_prob = tf.reduce_logsumexp(forward_log_probs[-1], axis=-1)

              def _scan_multiple_steps_backwards():
                """Perform `scan` operation when `num_steps` > 1."""

                def backward_step(log_previous_step, log_prob_observation):
                  return _log_matrix_vector(
                      log_transition,
                      log_prob_observation + log_previous_step)

                backward_log_adjoint_probs = tf.scan(
                    backward_step,
                    observation_log_probs[1:],
                    initializer=log_adjoint_prob,
                    reverse=True,
                    name="backward_log_adjoint_probs")

                return tf.concat([backward_log_adjoint_probs,
                                  [log_adjoint_prob]], axis=0)

              backward_log_adjoint_probs = prefer_static.cond(
                  self._num_steps > 1,
                  _scan_multiple_steps_backwards,
                  lambda: tf.convert_to_tensor([log_adjoint_prob]))

              backward_log_adjoint_probs = tf.Variable(backward_log_adjoint_probs)

              log_likelihoods = forward_log_probs + backward_log_adjoint_probs

              marginal_log_probs = distribution_util.move_dimension(
                  log_likelihoods - total_log_prob[..., tf.newaxis], 0, -2)

              self.forward_log_probs = forward_log_probs
              self.backward_log_probs = backward_log_adjoint_probs # ends with 0s
              self.fb_array = marginal_log_probs # doesn't end with 0s

              return categorical.Categorical(logits=marginal_log_probs)


def _log_vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices assuming values stored are logs."""
  return tf.reduce_logsumexp(vs[..., tf.newaxis] + ms, axis=-2)


def _log_matrix_vector(ms, vs):
  """Multiply tensor of matrices by vectors assuming values stored are logs."""
  return tf.reduce_logsumexp(ms + vs[..., tf.newaxis, :], axis=-1)
