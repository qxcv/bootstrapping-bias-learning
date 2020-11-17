import jax
import jax.numpy as jnp
import jax.random as jrandom

from pref_bootstrap.feedback_learner_base import EnvFeedbackModel


class PairedCompFeedbackModel(EnvFeedbackModel):
    """Feedback model for Boltzmann-rational paired comparisons."""
    def init_bias_params(self, rng):
        # sample from log-normal distribution
        rng_in, rng_out = jrandom.split(rng)
        # I believe the sole parameter here is the 'alpha' parameter for the
        # distribution
        params = jrandom.gamma(rng_in, 1.0, shape=())
        return params, rng_out

    def create_bias_prior(self, rng):
        rng_in, rng_out = jrandom.split(rng)
        # FIXME(sam): add a log-normal or gamma prior, and projection function
        # that clips to positive numbers (maybe; you might also be able to come
        # up with a more sensible parameterisation which does not require that)
        return prior, rng_out

    def _make_reward_fn(self, reward_model):
        def fn(inputs):
            out_values = reward_model.out(inputs)
            return out_values
        return fn

    def _compute_comparison_diffs(self, data, obs_fn):
        """Common pattern for all log likelihood/gradient methods in this class:

        1. Apply a function of observations to each observation in the
           environment.
        2. Use that precomputed table of values to look up value of the
           function for each observation in each supplied trajectory.
        3. Sum over the time axis, to make obs_fn a function if trajectories
           rather than just single observations (e.g. computing return from
           rewards at each time step).
        4. Using each supplied comparison pair `tau1 >= tau2`, compute
           `fn(tau1) - fn(tau2)`.
        5. Return just those differences.

        This same pattern is used to compute both return differences and reward
        gradient differences."""

        # first compute all values
        all_fn_vals = obs_fn(self.env.observation_matrix)

        # now use precomputed values to evaluate the function at each
        # observation
        trajectories = data['trajectories']
        states = trajectories['states']
        flat_states = states.flatten()
        flat_fn_vals = all_fn_vals[flat_states]

        # shape back into normal shape & sum over time axis
        per_obs_vals = jnp.reshape(
            flat_fn_vals, states.shape[:2] + all_fn_vals.shape[1:])
        per_traj_vals = jnp.sum(per_obs_vals, axis=1)

        # extract comparisons
        comparisons = data['comparisons']
        better_traj_ids = comparisons[:, 0]
        worse_traj_ids = comparisons[:, 1]

        # now return differences
        diffs = per_traj_vals[better_traj_ids] - per_traj_vals[worse_traj_ids]

        return diffs

    def log_likelihood(self, data, reward_model, bias_params):
        assert bias_params.ndim == 0, bias_params.shape
        ret_diffs = self._compute_comparison_diffs(
            data, self._make_reward_fn(reward_model))
        temp_diffs = bias_params * ret_diffs  # multiplicative temperature
        assert temp_diffs.shape == ret_diffs.shape
        log_likelihoods = jax.nn.log_sigmoid(temp_diffs)

        # average over trajectories (i.e. expected log likelihood, with
        # expectation taken w.r.t. empirical distribution and log likelihood
        # taken w.r.t. our model)
        return jnp.mean(log_likelihoods)

    def log_likelihood_grad_rew(self, data, reward_model, bias_params):
        ret_grad_diffs = self._compute_comparison_diffs(
            data, reward_model.grads)
        ret_diffs = self._compute_comparison_diffs(
            data, self._make_reward_fn(reward_model))
        grad_temps = bias_params * ret_grad_diffs
        temps = bias_params * ret_diffs
        grad_scales = 1 - jax.nn.sigmoid(temps)
        all_comparison_grads = grad_temps * grad_scales[:, None]

        # trajectory averaging again
        return jnp.mean(all_comparison_grads, axis=0)

    def log_likelihood_grad_bias(self, data, reward_model, bias_params):
        ret_diffs = self._compute_comparison_diffs(
            data, self._make_reward_fn(reward_model))
        temps = bias_params * ret_diffs
        all_beta_grads = ret_diffs * (1 - jax.nn.sigmoid(temps))

        # again average over trajectories
        return jnp.mean(all_beta_grads, axis=0)
