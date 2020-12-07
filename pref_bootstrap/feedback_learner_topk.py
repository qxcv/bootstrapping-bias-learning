import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, value_and_grad

from pref_bootstrap.feedback_learner_base import EnvFeedbackModel

from pref_bootstrap import priors

from jax import jit


class TopKFeedbackModel(EnvFeedbackModel):
    """Feedback model for Boltzmann-rational paired comparisons."""
    def __init__(self, env):
        super().__init__(env)
        self._bias_prior = priors.MixedPrior(lam=(1.0), mean=15, std=6.0)
    
    @property
    def bias_prior(self):
        return self._bias_prior    

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


    def log_likelihood(self, data, reward_model, bias):
        reward_model = reward_model.get_params()
        return self.loss(bias, reward_model, data)
        

    def log_likelihood_grad_rew(self, data, reward_model, bias_params):
        reward_model = reward_model.get_params()
        grads = grad(self.loss, 1)(bias_params, reward_model, data)
        return grads

    def log_likelihood_grad_bias(self, data, reward_model, bias_params):
        reward_model = reward_model.get_params()
        grads = grad(self.loss, 0)(bias_params, reward_model, data)
        return grads

    def loss(self, params, rmodel, data):
        targets = data['labels']
        inputs = data['trajectories']
        preds = self.predict(rmodel, params, inputs)
        label_probs = preds*targets + (1-preds)*(1-targets)
        return jnp.mean(jnp.log(label_probs+1e-12))
    
    def normal_grad(self, data, reward_model, bias_params):
        reward_model = reward_model.get_params()
        loss, grads = value_and_grad(self.loss, (0, 1))(bias_params, reward_model, data)
        return loss, grads
        
    def predict(self, rmodel, params, states): 
        """takes in: parameters"""
        flat_states = states.flatten()
        all_fn_values = rmodel #(self.env.observation_matrix)
        rew_est = (all_fn_values[flat_states]) # hopefully jax can do this, if not...need 1-hot.
        per_obs_rew  = jnp.reshape(rew_est, states.shape[:2] + rew_est.shape[1:])
        per_traj_rew_est = jnp.sum(per_obs_rew, axis=1)
        return jax.nn.sigmoid(params[0]*(per_traj_rew_est-params[1]))
    
    #TODO write a training function for this. 
    