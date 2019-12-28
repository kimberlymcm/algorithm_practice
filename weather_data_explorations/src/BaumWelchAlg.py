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

    def __init__(self, initial_distribution, observation_distribution, transition_distribution, num_steps, epsilon = 0.001, maxStep = 10):

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
        observation_log_probs = self._observation_log_probs(x, mask=None)     
        with tf.name_scope('forward_belief_propagation'):
            # forward belief propagation
            print("BEFORE")
            print(self.forward_log_probs)
            self._forward(observation_log_probs)
            print("AFTER")
            print(self.forward_log_probs)

        obs_prob_seq_rev = tf.reverse(obs_prob_seq, [True, False])
        obs_prob_list_back = tf.split(obs_prob_seq_rev, self.N, 0)

        with tf.name_scope('backward_belief_propagation'):
            # backward belief propagation
            self._backward(observation_log_probs)
        print(self.forward_log_probs)
        print(self.backward_log_probs)
        print("GOT THROUGH FORWARD BACKWARD")


    def _forward(self, observation_log_probs):
        print("In forward!")
    
        with tf.name_scope('forward_first_step'):
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
        # KM NEED TO FIX THIS FOR LOGS
        # import pdb; pdb.set_trace()
        
        states_marginal = tf.reduce_sum(self.gamma, 0)
        # 0s matrix, (1886, 2)
        seq_one_hot = tf.cast(tf.one_hot(tf.cast(x, tf.int64), self.O, 1, 0), tf.float32)
        emission_score = tf.reduce_logsumexp(seq_one_hot + self.gamma, axis=-2)
        #emission_score = tf.matmul(tf.cast(seq_one_hot, tf.float32), self.gamma, transpose_a=True)

        # x, 
        emission_score = tf.math.reduce_sum(tf.multiply(tf.math.exp(self.fb_array), tf.expand_dims(x, 1)), 0)
        new_means = emission_score / states_marginal

        new_variances = []
        for i in range(self.O):
            tmp_0 = (x - new_means[i])**2
            tmp_1 = tf.tensordot(tf.math.exp(self.fb_array[:, i]), tmp_0, axes=1)
            #tmp_1 = tf.math.reduce_sum(tf.multiply(tf.expand_dims(tf.transpose(tf.math.exp(self.fb_array[:, i]))), tf.expand_dims(tmp_0, 1)), 0)
            new_var = tmp_1 / np.array(states_marginal)[i]
            new_variances.append(new_var)

        new_emissions = tfp.distributions.Normal(loc=np.array(new_means), scale=new_variances)

        return new_emissions


    def re_estimate_transition(self, x, fb_array):
        
        with tf.name_scope('Init_3D_tensor'):
            # What is M??
            # M: v tensor (pg 70): num_states x num_states x num_observations
            self.M = tf.Variable(tf.zeros((self.N-1, self.S, self.S), tf.float32))
        
        with tf.name_scope('3D_tensor_transition'):
            # For each observation
            for t in range(self.N - 1):
                with tf.name_scope('time_step-%s' %t):
                    # tmp_0 = prob of being in state * prob of transitioning out of state
                    tmp_00 = tf.expand_dims(self.forward_log_probs[t, :], 0)
                    tmp_0 = _log_vector_matrix(tmp_00, self.transition_distribution.logits_parameter())
                    #tmp_0 = tf.matmul(tf.expand_dims(self.forward_log_probs[t, :], 0), self.transition_distribution)
                    #tmp_0 = tf.matmul(tf.expand_dims(fb_array[t, :], 0), self.T)
                    # tmp_1 = prob_in_state_at_t * p_transitioning_to_state_y_at_t+1 * p_emitting_d_at_x+1_given_observed_state
                    # KM I AM STUCK HERE. THE OBSERVATION_DISTRIBUTION is a NORMAL distribution,so I need to get those,
                    # and then do the addition
                    #import pdb; pdb.set_trace()

                    # tmp_1 = tf.multiply(tmp_0, tf.expand_dims(tf.gather(self.observation_distribution.logits_parameter(), x[t+1]), 0))
                    # My attempt below
                    tmp_1 = tmp_0 + self._observation_distribution.log_prob(x[t+1])
                    denom = tf.squeeze(_log_vector_matrix(tmp_1, self.backward_log_probs[t+1, :]))
                    #print("DENOM")
                    #print(denom)
                    #denom = tf.squeeze(tf.matmul(tmp_1, tf.expand_dims(self.backward_log_probs[t+1, :], 1)))

                with tf.name_scope('Init_new_transition'):
                    trans_re_estimate = tf.Variable(tf.zeros((self.S, self.S), tf.float32))
                
                # For each state
                # KM: This is what will need to be changed for the normal distribution
                # Need to get the u matrix
                #u_mat = tf.multiply(self.forward, self.backward) 
                
                # For each state:
                #     forward[t, i] * transition[1->0, 1->1] * observation[1|0, 1|0] * backward[t+1]
                for i in range(self.S):
                    with tf.name_scope('State-%s' %i):
                        # KM tried to fix
                        #import pdb; pdb.set_trace()
                        tmp_0 = self.forward_log_probs[t, i] + np.array(self.transition_distribution.logits_parameter())[i, :]
                        numer = tmp_0 + self._observation_distribution.log_prob(x[t+1]) + self.backward_log_probs[t+1, :]
                        #numer = self.forward[t, i] * self.T[i, :] * tf.gather(self.E, x[t+1]) * self.backward[t+1, :]
                        # DOES THIS NEED TO BE numer - denom????
                        trans_re_estimate = tf.compat.v1.scatter_update(trans_re_estimate, i, numer / denom)
                        #for t in range(self.N)

                self.M = tf.compat.v1.scatter_update(self.M, t, trans_re_estimate)

        # What is this doing
        with tf.name_scope('Smooth_gamma'):
            # tf.reduce_sum is adding the two log_probs, going from 3D to 2D
            # idk what squeeze does
            self.gamma = tf.squeeze(tf.reduce_sum(self.M, 2))
            # tf.reduce_sum(self.M, 0): Sums rows and then sums colomns, so it is 2x2
            # tf.reduce_sum(self.gamma, 0): Sums rows so it is a 2x1
            T_new = tf.reduce_sum(self.M, 0) / tf.expand_dims(tf.reduce_sum(self.gamma, 0), 1)
            T_new = tfp.distributions.Categorical(T_new)
        
        with tf.name_scope('New_init_states_prob'):
            T0_new = tfp.distributions.Categorical(probs=self.gamma[0,:])

        with tf.name_scope('Append_gamma_final_time_step'):
            #import pdb; pdb.set_trace()
            #prod = tf.expand_dims(tf.multiply(self.forward[self.N-1, :], self.backward[self.N-1, :]), 0)
            prod = self.forward_log_probs[self.N-1, :] + self.backward_log_probs[self.N-1, :]
            # THIS MIGHT NEED TO BE A SUBTRACTION TOO
            ss = prod / tf.reduce_sum(prod)
            #import pdb; pdb.set_trace()
            #self.gamma = tf.concat([np.array(self.gamma), np.array(ss)], 0)
            self.gamma = tf.compat.v1.concat([self.gamma, tf.expand_dims(ss, 0)], 0)

            self.prob_state_1.append(self.gamma[:, 0])
        
        return T0_new, T_new
    
    def check_convergence(self, new_T0, new_transition, new_emission):
        delta_T0 = tf.reduce_max(tf.abs(self.initial_distribution.logits_parameter() - new_T0.logits_parameter())) < self.epsilon
        delta_T = tf.reduce_max(tf.abs(self.transition_distribution.logits_parameter() - new_transition.logits_parameter())) < self.epsilon
        delta_E = tf.reduce_max(tf.abs(self.observation_distribution.mean() - new_emission.mean())) < self.epsilon

        return tf.logical_and(tf.logical_and(delta_T0, delta_T), delta_E)
        


    def expectation_maximization_step(self, x): # x is the observations
        
        # probability of emission sequence
        #obs_prob_seq = tf.gather(self.E, x) 
        #obs_prob_seq = tf.gather(self.observation_distribution, x)

        # Hopefully this is the log probability of the observed sequence
        #import pdb; pdb.set_trace()

        obs_prob_seq = tf.math.exp(self._observation_log_probs(x, mask=None)) # NOT LOGS
        print(obs_prob_seq)

        ## probability of emission sequence
        with tf.name_scope('Forward_Backward'):
            print("above to run forward_backward")
            self.forward_backward(obs_prob_seq, x)
            self.fb_array = self.forward_log_probs + self.backward_log_probs
            # Not sure what is happening here
            # Not sure if I need to do this
            total_log_prob = tf.reduce_logsumexp(self.forward_log_probs[-1], axis=-1)
            marginal_log_probs = distribution_util.move_dimension(self.fb_array - total_log_prob[..., tf.newaxis], 0, -2)
            self.fb_array = marginal_log_probs
        #obs_prob_seq = tf.gather(self.E, x)
        #with tf.name_scope('Forward_Backward'):
        #    self.forward_backward(obs_prob_seq)

        #with tf.name_scope('Forward_Backward'):
            #self.forward_backward(obs_prob_seq)
            # I guess I don't wanna actually store these
            # need to get out the forward - bback
            #fb = self.posterior_marginals(x, mask=None, name=None)
            ## These are the forward / backward probabilities
            #fb_array = np.array(fb.prob_parameter())
            #print(fb_array)
            #sys.exit()

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
            self._transition_distribution = new_transition
            #self.T0 = tf.assign(self.T0, new_T0)
            #self.E = tf.assign(self.E, new_emission)
            #self.T = tf.assign(self.T, new_transition)
            #self.count = tf.assign_add(self.count, 1)
             
            #with tf.name_scope('histogram_summary'):
            #    _ = tf.summary.histogram(self.initial_distribution)
            #    _ = tf.summary.histogram(self.transition_distribution)
            #    _ = tf.summary.histogram(self.observation_distribution)
        return converged
        
 
    def Baum_Welch_EM(self, obs_seq):
        
        with tf.name_scope('Input_Observed_Sequence'):
            # length of observed sequence
            #self.N = len(obs_seq)
            self.N = len(np.array(obs_seq)) 

            # shape of Variables
            shape = [self.N, self.S]

            # observed sequence
            x = tf.constant(obs_seq, dtype=tf.float32, name='observation_sequence')
        
        converged = tf.cast(False, tf.bool)
        #self.count = tf.Variable(tf.constant(0))
        
        with tf.name_scope('Train_Baum_Welch'):
            for i in range(self.maxStep): # Loop through each step
                
                with tf.name_scope('EM_step-%s' %i):
                    converged = self.expectation_maximization_step(x)
                print("II")
                print(self._initial_distribution)

      
        return converged
    
    def run_Baum_Welch_EM(self, obs_seq, summary=False, monitor_state_1=False):

        observations = tf.convert_to_tensor(obs_seq)
        observations = tf.cast(observations, tf.float32)
        
        fb_array = self.posterior_marginals(observations, mask=None, name=None)
        
        converged = self.Baum_Welch_EM(obs_seq)
        
        # Summary
        #summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            trans0, transition, emission, c = sess.run([self.T0, self.T, self.E, converged])
            
            #if monitor_state_1:
            #    self.state_summary = np.array([sess.run(g) for g in self.prob_state_1])
            
            #if summary:
                # Instantiate a FileWriter to output summaries and the Graph.
            #    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
            #    summary_str = sess.run(summary_op)
            #    summary_writer.add_summary(summary_str)

            return trans0, transition, emission, c

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
              self.backward_log_probs = backward_log_adjoint_probs
              self.fb_array = marginal_log_probs

              return categorical.Categorical(logits=marginal_log_probs)


def _log_vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices assuming values stored are logs."""
  return tf.reduce_logsumexp(vs[..., tf.newaxis] + ms, axis=-2)


def _log_matrix_vector(ms, vs):
  """Multiply tensor of matrices by vectors assuming values stored are logs."""
  return tf.reduce_logsumexp(ms + vs[..., tf.newaxis, :], axis=-1)


def _vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices."""
  return tf.reduce_sum(vs[..., tf.newaxis] * ms, axis=-2)


def _extract_log_probs(num_states, dist):
  """Tabulate log probabilities from a batch of distributions."""
  states = tf.reshape(tf.range(num_states),
                      tf.concat([[num_states],
                                 tf.ones_like(dist.batch_shape_tensor())],
                                axis=0))
  return distribution_util.move_dimension(dist.log_prob(states), 0, -1)