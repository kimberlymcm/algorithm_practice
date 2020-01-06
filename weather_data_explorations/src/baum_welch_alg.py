import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static


class BaumWelch(tfp.distributions.HiddenMarkovModel):
    """
    Extends the tensorflow HiddenMarkoModel class by implementing the BaumWelch ALgorithm
    """

    def __init__(
            self,
            initial_distribution,
            observation_distribution,
            transition_distribution,
            num_steps,
            epsilon=0.05,
            maxStep=10,
            log_dir='.'):

        super().__init__(
            initial_distribution,
            transition_distribution,
            observation_distribution,
            num_steps)

        with tf.name_scope('Inital_Parameters'):
            with tf.name_scope('Scalar_constants'):

                self.logdir = log_dir

                # Max number of iteration
                self.maxStep = maxStep

                # convergence criteria
                self.epsilon = epsilon

                # Number of possible states
                self.S = np.int(self.initial_distribution.num_categories)

                self.forward_log_probs = tf.Variable(
                    tf.zeros(self.S, dtype=tf.float64), name='forward')
                self.backward_log_probs = tf.Variable(
                    tf.zeros(self.S, dtype=tf.float64), name='backward')
                self.fb_array = tf.Variable(
                    tf.zeros(
                        self.S,
                        dtype=tf.float64),
                    name='posterior')


    def forward_backward(self, x):
        """
        runs forward backward algorithm on observation sequence
        Arguments
        ---------
        - x : array of size N with the observations
        Returns Nothing
        -------
        Updates
        - forward_log_probs : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward_log_probs : matrix of size N by S representing
            the backward probability of each state at each time step
        """
        observation_log_probs = self._observation_log_probs(x, mask=None)
        with tf.name_scope('forward_belief_propagation'):
            self.forward_log_probs = self._forward(observation_log_probs)

        with tf.name_scope('backward_belief_propagation'):
            self.backward_log_probs = self._backward(observation_log_probs)


    def _forward(self, observation_log_probs):

        with tf.name_scope('forward_step'):
            # prob_starting_in_each_state * prob_of_observed_state_given_hiddenstate[0]
            init_prob = self.initial_distribution.logits_parameter() + \
                tf.squeeze(observation_log_probs[0])
            log_transition = self.transition_distribution.logits_parameter()

            def _scan_multiple_steps_forwards():
                def forward_step(log_previous_step, log_prob_observation):
                    return _log_vector_matrix(
                        log_previous_step, log_transition) + log_prob_observation

                forward_log_probs = tf.scan(forward_step,
                                            observation_log_probs[1:],
                                            initializer=init_prob,
                                            name="forward_log_probs")
                return tf.concat([[init_prob], forward_log_probs], axis=0)

            forward_log_probs = prefer_static.cond(
                self._num_steps > 1,
                _scan_multiple_steps_forwards,
                lambda: tf.convert_to_tensor([init_prob]))

        return forward_log_probs

    def _backward(self, observation_log_probs):

        with tf.name_scope('backward_step'):
            init_prob = self.initial_distribution.logits_parameter() + \
                tf.squeeze(observation_log_probs[0])
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
        return backward_log_adjoint_probs


    def re_estimate_emission(self, x):
        """
        Updates the observation_distribution, assuming it is a normal dist 
        Arguments
        ---------
        - x : array of size N with the observations
        Returns
        _________
        - new_emissions: Normal dist with new inferred emissions
        """
        with tf.name_scope('update_emissions'):
            u_x = tf.multiply(tf.math.exp(self.fb_array), tf.expand_dims(x, 1)) # pg 73: uj(t)*x(t)
            emission_score_log = tf.math.log(tf.math.reduce_sum(u_x, 0))

            denom = tf.math.reduce_logsumexp(self.fb_array, 0)
            means_log = emission_score_log - denom
            means = tf.math.exp(means_log)

            new_stds = []
            for i in range(self.S):
                variance_array = (x - tf.math.exp(means_log[i]))**2
                variance_array_x = tf.multiply(tf.math.exp(
                    self.fb_array[:, i]), variance_array)  # not logs
                variance_score = tf.math.reduce_sum(variance_array_x, 0)
                new_var = variance_score / tf.math.exp(denom[i])
                new_std = tf.math.sqrt(new_var)
                new_stds.append(new_std)

            new_emissions = tfp.distributions.Normal(loc=means, scale=new_std)

        return new_emissions


    def re_estimate_transition(self, x, fb_array):

        with tf.name_scope('Init_3D_tensor'):
            # M: v tensor (pg 70): num_states x num_states x num_observations
            # u (pg 70) is the fb_array I think
            self.M = tf.Variable(
                tf.zeros(
                    (self.N - 1, self.S, self.S), tf.float32))

            total_log_prob = tf.reduce_logsumexp(
                self.forward_log_probs[-1], axis=-1)

        with tf.name_scope('3D_tensor_transition'):

            for t in range(self.N - 1):
                with tf.name_scope('time_step-%s' % t):
                    # tmp_0 = prob of being in state * prob of transitioning out of state
                    # eq 4.14 pg 71
                    forward_trans = _log_vector_matrix(
                        tf.expand_dims(self.forward_log_probs[t, :], 0),
                        self.transition_distribution.logits_parameter())

                    forward_trans_obs = forward_trans + \
                        self._observation_distribution.log_prob(x[t + 1])

                    denom = tf.squeeze(_log_vector_matrix(
                        forward_trans_obs, self.backward_log_probs[t + 1, :]))

                with tf.name_scope('Init_new_transition'):
                    trans_re_estimate = tf.Variable(
                        tf.zeros((self.S, self.S), tf.float32))

                # For each state:
                # forward[t, i] * transition[1->0, 1->1] * observation[1|0,
                # 1|0] * backward[t+1]
                for i in range(self.S):
                    with tf.name_scope('State-%s' % i):
                        # P(being in state i at time t) * [P(trans i->j),
                        # P(i->i)]
                        tmp_0 = self.forward_log_probs[t, i] + np.array(
                            self.transition_distribution.logits_parameter())[i, :]
                        # [P(state i time t & [P(i->i), P(i->i)]) * [P(o|i), P[o|j]]
                        # [P(state i time t & P(i->i) & P(o|i) & P(state i time t+1))]
                        # [P(state i time t & P(i->j) & P(o|j) & P(state ij time t+1))]
                        numer = tmp_0 + \
                            self._observation_distribution.log_prob(x[t + 1]) + self.backward_log_probs[t + 1, :]
                        trans_re_estimate = tf.compat.v1.scatter_update(
                            trans_re_estimate, i, numer - total_log_prob)

                self.M = tf.compat.v1.scatter_update(
                    self.M, t, trans_re_estimate)

        with tf.name_scope('Smooth_gamma'):

            # 4x4, need to do this to get sums of logs
            numer = tf.reduce_logsumexp(self.M, 0)
            # Might need to be over axis 1, not 0
            denom = tf.reduce_logsumexp(numer, 0)
            T_new_tmp = numer - denom
            T_new = tfp.distributions.Categorical(logits=T_new_tmp)

        with tf.name_scope('New_init_states_prob'):
            T0_new = tfp.distributions.Categorical(
                logits=self.fb_array[0, :])  # 1 pg 72

        return T0_new, T_new

    def check_convergence(self, new_T0, new_transition, new_emission):
        # TODO (kmcmanus): Log output to a file
        delta_T0 = tf.reduce_max(
            tf.abs(
                self.initial_distribution.probs_parameter() -
                new_T0.probs_parameter()) /
            self.initial_distribution.probs_parameter())
        delta_T0_less = delta_T0 < self.epsilon
        print(delta_T0)

        delta_T = tf.reduce_max(
            tf.abs(
                self.transition_distribution.probs_parameter() -
                new_transition.probs_parameter()) /
            self.transition_distribution.probs_parameter())
        delta_T_less = delta_T < self.epsilon

        delta_E = tf.reduce_max(
            tf.abs(
                self.observation_distribution.mean() - new_emission.mean()
            ) / self.observation_distribution.mean()
        ) 
        delta_E_less = delta_E < self.epsilon

        #return tf.logical_and(tf.logical_and(delta_T0_less, delta_T_less), delta_E_less)
        return delta_T0 + delta_T + delta_E


    def expectation_maximization_step(self, x):

        obs_prob_seq = tf.math.exp(
            self._observation_log_probs(
                x, mask=None))  # NOT LOGS

        # probability of emission sequence
        with tf.name_scope('Forward_Backward'):
            self.forward_backward(x)

            total_log_prob = tf.reduce_logsumexp(
                self.forward_log_probs[-1], axis=-1)
            log_likelihoods = self.forward_log_probs + self.backward_log_probs
            marginal_log_probs = distribution_util.move_dimension(
                log_likelihoods - total_log_prob[..., tf.newaxis], 0, -2)
            self.fb_array = marginal_log_probs

        with tf.name_scope('Re_estimate_transition'):
            new_T0, new_transition = self.re_estimate_transition(
                x, self.fb_array)

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
            converged = self.check_convergence(
                new_T0, new_transition, new_emission)

        with tf.name_scope('Update_parameters'):
            #self.initial_distribution = tf.compat.v1.assign(self.initial_distribution, new_T0)
            #self.observation_distribution = tf.compat.v1.assign(self.observation_distribution, new_emission)
            #self.transition_distribution = tf.compat.v1.assign(self.transition_distribution, new_transition)
            self._initial_distribution = new_T0
            self._observation_distribution = new_emission
            self._transition_distribution = new_transition

        return converged

    def run_Baum_Welch_EM(
            self,
            observations,
            summary=False,
            monitor_state_1=False):

        with tf.name_scope('Train_Baum_Welch'):

            self.N = len(np.array(observations))
            # converged = tf.cast(False, tf.bool)
            converged = tf.Variable(0)

            #change_summary = tf.summary.scalar('change_sum', converged)
            summary_writer = tf.summary.create_file_writer(self.logdir)
            #init = tf.global_variables_initializer()

            with summary_writer.as_default():
                with tf.name_scope('EM_steps'):
                    for i in range(self.maxStep):
                        converged = self.expectation_maximization_step(observations)
                        tf.summary.scalar('change_sum', converged, step=i)
                        if converged < self.epsilon:
                            print("CONVERGED")
                            break
                    #filewriter.close()

        if converged:
            print("Algorithm Converged")

        return self.initial_distribution, self.transition_distribution, self.observation_distribution


def _log_vector_matrix(vs, ms):
    """Multiply tensor of vectors by matrices assuming values stored are logs."""
    return tf.reduce_logsumexp(vs[..., tf.newaxis] + ms, axis=-2)


def _log_matrix_vector(ms, vs):
    """Multiply tensor of matrices by vectors assuming values stored are logs."""
    return tf.reduce_logsumexp(ms + vs[..., tf.newaxis, :], axis=-1)
