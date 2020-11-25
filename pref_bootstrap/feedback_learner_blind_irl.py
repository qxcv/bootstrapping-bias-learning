"""Implementation of EnvFeedbackModel for a sub-rational variant of MCE IRL
where the demonstrator is modelled as being 'blind' to some elements of the
observation vector."""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from pref_bootstrap.algos.mce_irl import mce_occupancy_measures, mce_partition_fh
from pref_bootstrap.feedback_learner_base import EnvFeedbackModel


class BlindIRLFeedbackModel(EnvFeedbackModel):
    """Feedback model for MCE IRL, with sub-rationality coming from the agent
    being "blind" to certain elements of the state."""

    def init_bias_params(self, rng):
        # WARNING: these bias params are meant to be constrained to 0<=b<=1 (or
        # some other convex approximation of {0,1}).
        rng_in, rng_out = jrandom.split(rng)
        obs_dim = self.env.obs_dim
        params = jrandom.beta(key=rng_in, a=0.5, b=0.5, shape=(obs_dim,))
        return params, rng_out

    def create_bias_prior(self, rng):
        rng_in, rng_out = jrandom.split(rng)
        # FIXME(sam): add a beta prior, and also improve the prior API so that
        # you can project into support
        return prior, rng_out

    def log_likelihood(self, data, reward_model, bias_params):
        # expected log likelihood of some trajectories under the current reward
        # model
        assert isinstance(data, dict)
        assert {"states", "acts"} <= data.keys()
        states, acts = data["states"], data["acts"]
        B, T = acts.shape
        assert states.shape == (B, T + 1)

        # compute log probability of initial states
        init_states = states[:, 0]
        init_state_probs = self.env.initial_state_dist[init_states]
        assert init_state_probs.shape == (len(init_states),)
        # average over dataset to compute expectation
        log_init_probs = jnp.mean(jnp.log(init_state_probs))

        # compute log probability of state transitions
        states_t_flat = states[:, :-1].flatten()
        states_tp1_flat = states[:, 1:].flatten()
        acts_flat = acts.flatten()
        trans_idx_tup = (states_t_flat, acts_flat, states_tp1_flat)
        trans_probs = self.env.transition_matrix[trans_idx_tup]
        trans_probs = trans_probs.reshape((B, T))
        # sum over time, average over dataset
        log_trans_probs = jnp.mean(jnp.sum(jnp.log(trans_probs), axis=1), axis=0)

        # compute log probability of actions under the current model
        # first compute 'blinded' observations
        blind_obs_mat = self.env.observation_matrix * bias_params[None]
        assert blind_obs_mat.shape == self.env.observation_matrix.shape
        blind_reward_mat = reward_model.out(blind_obs_mat)
        V, Q, pi = mce_partition_fh(self.env, R=blind_reward_mat)
        # pi is indexed by pi[t,s,a]
        time_indices = jnp.repeat(jnp.arange(T)[None, :], B, axis=0)
        time_indices_flat = time_indices.flatten()
        pi_idx_tup = (time_indices_flat, states_t_flat, acts_flat)
        act_probs = pi[pi_idx_tup]
        act_probs = act_probs.reshape((B, T))
        # sum over time, average over dataset
        log_act_probs = jnp.mean(jnp.sum(jnp.log(act_probs), axis=1), axis=0)

        # compute log probability is sum of log probs
        log_prob = log_init_probs + log_trans_probs + log_act_probs

        return log_prob

    def _ll_compute_oms(self, data, reward_model, bias_params):
        # 'blind' the observations to compute obs mat
        blind_obs_mat = self.env.observation_matrix * bias_params[None]
        assert blind_obs_mat.shape == self.env.observation_matrix.shape
        states = data["states"]
        states_t = states[:, :-1]
        T = states_t.shape[1]

        # compute exact occupancy measure for a policy optimal w.r.t. those
        # 'blind' observations
        blind_rews = reward_model.out(blind_obs_mat)
        om_t, om = mce_occupancy_measures(self.env, R=blind_rews)
        assert om.shape == (self.env.n_states,)
        assert om_t.shape == (T, self.env.n_states)

        # now compute empirical occupancy measure
        state_eye = jnp.eye(self.env.n_states)
        empirical_om_flat = state_eye[states_t.flatten()]
        om_t_shape = states_t.shape + (self.env.n_states,)
        empirical_om_t = jnp.mean(empirical_om_flat.reshape(om_t_shape), axis=0)
        assert empirical_om_t.shape == om_t.shape
        # sum over time axis
        empirical_om = jnp.sum(empirical_om_t, axis=0)

        return om, empirical_om, blind_obs_mat

    def log_likelihood_grad_rew(self, data, reward_model, bias_params):
        # This function computes this difference:
        #     E_D[\nabla_theta R_theta(bias*obs)]
        #         - E_pi[\nabla_\theta R_\theta(bias*obs)]
        om, empirical_om, blind_obs_mat = self._ll_compute_oms(
            data, reward_model, bias_params
        )

        # compute reward gradient in each state
        reward_grads = reward_model.grads(blind_obs_mat)

        empirical_grad_term = jnp.mean(empirical_om[:, None] * reward_grads, axis=0)
        pi_grad_term = jnp.mean(om[:, None] * reward_grads, axis=0)
        grads = empirical_grad_term - pi_grad_term

        return grads

    def log_likelihood_grad_bias(self, data, reward_model, bias_params):
        om, empirical_om, _ = self._ll_compute_oms(data, reward_model, bias_params)

        def blind_reward(biases, obs_matrix):
            """Compute blind reward for all states in such a way that Jax can
            differentiate with respect to the bias/masking vector. This is
            trivial for linear rewards, but harder for more general
            RewardModels."""
            blind_obs_mat = obs_matrix * biases
            assert blind_obs_mat.shape == obs_matrix.shape
            return reward_model.out(blind_obs_mat)

        # compute gradient of reward in each state w.r.t. biases
        # (we do this separately for each input)
        blind_rew_grad_fn = jax.grad(blind_reward)
        lifted_blind_rew_grad_fn = jax.vmap(jax.partial(blind_rew_grad_fn, bias_params))
        lifted_grads = lifted_blind_rew_grad_fn(self.env.observation_matrix)

        empirical_grad_term = jnp.mean(empirical_om[:, None] * lifted_grads, axis=0)
        pi_grad_term = jnp.mean(om[:, None] * lifted_grads, axis=0)
        grads = empirical_grad_term - pi_grad_term

        return grads
