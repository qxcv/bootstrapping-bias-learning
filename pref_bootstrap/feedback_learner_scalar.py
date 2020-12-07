import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from pref_bootstrap.feedback_learner_base import EnvFeedbackModel
from pref_bootstrap import priors

class ScalarFeedbackModel(EnvFeedbackModel):
    """Scalar Feedback model"""
    def __init__(self, env):
        super().__init__(env)
        self._bias_prior = priors.FixedGaussianPrior(shape=(2,))

    @property
    def bias_prior(self):
        return self._bias_prior

    def _make_reward_fn(self, reward_model):
        def fn(inputs):
            out_values = reward_model.out(inputs)
            return out_values

        return fn

    def log_likelihood(self, data, reward_model, bias_params):
        """Compute log likelihood of given human data under the current reward
        and bias model parameters."""
        states, acts, samples = data["trajectories"]["states"], data["trajectories"]["acts"], data["corrupted_rewards"]
        states = states.flatten()
        samples = samples.flatten()
        amount_of_samples = states.shape[0]
        obs_values = self._make_reward_fn(reward_model)
        all_vals = obs_values(self.env.observation_matrix)
        re_vals = all_vals[states]
        ll = ((-1 * amount_of_samples)/2) * np.log(2 * np.pi) - ((-1 * amount_of_samples)/2) * np.log(bias_params[1]) - \
            (1/(2* bias_params[1])) * np.sum((samples - re_vals - bias_params[0])**2)
        return ll
        
    def log_likelihood_grad_rew(self, data, reward_model, bias_params):
        """Compute gradient of log likelihood of human data with respect to
        reward parameters only."""
        obs, states, acts, samples = data["trajectories"]["obs"], data["trajectories"]["states"], data["trajectories"]["acts"], data["corrupted_rewards"]
        states = states.flatten()
        samples = samples.flatten()
        amount_of_samples = states.shape[0]
        obs_values = self._make_reward_fn(reward_model)
        all_vals = obs_values(self.env.observation_matrix)
        re_vals = all_vals[states]
        obs = obs.reshape((amount_of_samples, self.env.obs_dim))
        grad_rew = np.zeros(self.env.obs_dim)
        for i in range(amount_of_samples):
            grad_rew += obs[i] * (samples[i] - re_vals[i] - bias_params[0])
        grad_rew *= 1/(bias_params[1])
        return grad_rew

    def log_likelihood_grad_bias(self, data, reward_model, bias_params):
        states, acts, samples = data["trajectories"]["states"], data["trajectories"]["acts"], data["corrupted_rewards"]
        states = states.flatten()
        samples = samples.flatten()
        amount_of_samples = states.shape[0]
        obs_values = self._make_reward_fn(reward_model)
        all_vals = obs_values(self.env.observation_matrix)
        re_vals = all_vals[states]
        mu_grad = (1/bias_params[1]) * np.sum((samples - re_vals - bias_params[0])**2)
        sigma_grad = (1/(2* bias_params[1])) * ((-1 * amount_of_samples) + (1/bias_params[1]) *\
            np.sum((samples - re_vals - bias_params[0])**2))
        return np.array([mu_grad, sigma_grad])