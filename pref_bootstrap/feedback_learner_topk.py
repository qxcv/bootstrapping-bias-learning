import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, value_and_grad

from pref_bootstrap.feedback_learner_base import EnvFeedbackModel


class PairedCompFeedbackModel(EnvFeedbackModel):
    """Feedback model for Boltzmann-rational paired comparisons."""

    def init_bias_params(self, rng):
        # sample from log-normal distribution
        rng_in, rng_out = jrandom.split(rng)

        # TEMP SCALE: 
        temp_scale = 30
        min_temp = 10
        params = temp_scale*jrandom.uniform(rng_in, shape=()) + min_temp # Rand between 
        return params, rng_out

#     def create_bias_prior(self, rng):
#         rng_in, rng_out = jrandom.split(rng)
#         # FIXME(sam): add a log-normal or gamma prior, and projection function
#         # that clips to positive numbers (maybe; you might also be able to come
#         # up with a more sensible parameterisation which does not require that)
#         return prior, rng_out

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
        trajectories = data["trajectories"]
        states = trajectories["states"]
        flat_states = states.flatten()
        flat_fn_vals = all_fn_vals[flat_states]

        # shape back into normal shape & sum over time axis
        per_obs_vals = jnp.reshape(
            flat_fn_vals, states.shape[:2] + all_fn_vals.shape[1:]
        )
        per_traj_vals = jnp.sum(per_obs_vals, axis=1)

        # extract comparisons
        comparisons = data["comparisons"]
        better_traj_ids = comparisons[:, 0]
        worse_traj_ids = comparisons[:, 1]

        # now return differences
        diffs = per_traj_vals[better_traj_ids] - per_traj_vals[worse_traj_ids]

        return diffs

    def log_likelihood(self, data, labels, reward_model, bias_params):
#         assert bias_params.ndim == 0, bias_params.shape
#         ret_diffs = self._compute_comparison_diffs(
#             data, self._make_reward_fn(reward_model)
#         )
#         temp_diffs = bias_params * ret_diffs  # multiplicative temperature
#         assert temp_diffs.shape == ret_diffs.shape
#         log_likelihoods = jax.nn.log_sigmoid(temp_diffs)

#         # average over trajectories (i.e. expected log likelihood, with
#         # expectation taken w.r.t. empirical distribution and log likelihood
#         # taken w.r.t. our model)
#         return jnp.mean(log_likelihoods)
        all_rew_vals = obs_fn(self.env.observation_matrix)
        traj = data['trajectories']
        states = traj['states']
        flat_states = states.flatten()
        flat_fn_vals = all_fn_vals[flat_states]
        
        per_obs_vals = jnp.reshape(flat_fn_vals, states.shape[:2] + all_fn_vals.shape[1:])
        per_traj_vals = jnp.sum(per_obs_vals, axis=1)
        
        topk = data['topk'] # binary labels
        
        preds = self.predict(params, states)
        
        
        grad(loss, (0,1))(params['reward'], params['b'], params['temp'])
        
        
        
        

    def log_likelihood_grad_rew(self, data, reward_model, bias_params):

    def log_likelihood_grad_bias(self, data, reward_model, bias_params):


    def loss(self, params):
        preds = predict(params['reward_est'], params['temperature'], params['bias'], self.inputs)
        label_probs = preds*targets + (1-preds)*(1-targets)
        return -jnp.sum(jnp.log(label_probs))
        
        
        
    
    def predict(self, reward_est, temperature, bias, states): 
        
        """takes in: parameters, """
        flat_states = states.flatten()
        rew_est = (reward_est[flat_states]) # hopefully jax can do this, if not...need 1-hot.
        per_obs_rew  = jnp.reshape(rew_est, states.shape[:2] + rew_est.shape[1:])
        per_traj_rew_est = jnp.sum(per_obs_rew, axis=1)
        return jax.nn.sigmoid(-1*temperature*(per_traj_rew_est-bias))
    