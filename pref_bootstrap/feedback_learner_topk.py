import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, value_and_grad

from pref_bootstrap.feedback_learner_base import EnvFeedbackModel


class TopKFeedbackModel(EnvFeedbackModel):
    """Feedback model for Boltzmann-rational paired comparisons."""

    def init_bias_params(self, rng):
        # sample from log-normal distribution
        rng_in, rng_out = jrandom.split(rng)

        # TEMP SCALE: 
        temp_scale = .1
        min_temp = 0
        params = temp_scale*jrandom.uniform(rng_in, shape=()) + min_temp # Rand between 
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
        
        return topk*jnp.log(preds) + (1-topk)*jnp.log(1-preds)
        

    def log_likelihood_grad_rew(self,params, data):
        grads = grad(loss)(params, data)
        return grads['reward_est']

    def log_likelihood_grad_bias(self,params,data):
        grads = grad(loss)(params, data)
        return grads['b'], grads['temp']
        
    def grad_loss(self,params, data): 
        loss, grads = value_and_grad(loss(params, data))
        return loss, grads


    def loss(self, params, inputs, targets):
        preds = self.predict(params['reward_est'], params['temperature'], params['bias'], inputs)
        label_probs = preds*targets + (1-preds)*(1-targets)
        return -jnp.mean(jnp.log(label_probs+1e-12))
        
    def predict(self, reward_est, temperature, bias, states): 
        
        """takes in: parameters"""
        flat_states = states.flatten()
        rew_est = (reward_est[flat_states]) # hopefully jax can do this, if not...need 1-hot.
        per_obs_rew  = jnp.reshape(rew_est, states.shape[:2] + rew_est.shape[1:])
        per_traj_rew_est = jnp.sum(per_obs_rew, axis=1)
        return 1-jax.nn.sigmoid(temperature*(per_traj_rew_est-bias))
    
    
    #TODO write a training function for this. 
    