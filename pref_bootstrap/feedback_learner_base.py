"""Draft API for different feedback models."""

import abc


class EnvFeedbackModel(abc.ABC):
    """Abstract base class for a 'feedback model' in some environment.

    A feedback model is implicitly associated with a specific feedback modality
    (e.g. demonstrations, corrections, paired comparisons), and explicitly
    associated with a specific environment. `EnvFeedbackModel` instances do not
    have state of their own (except environment-specific metadata), but they
    are able to generate bias parameter vectors, as well as priors for those
    vectors. Later on, the bias parameter vector can be combined with a reward
    model to compute the log likelihood of some observed data, and to compute
    gradients of the log likelihood with respect to both reward parameters and
    bias parameters."""

    def __init__(self, env):
        self.env = env

    def init_bias_params(self, rng):
        """Generate a random set of bias model parameters for this feedback
        modality in this environment."""
        return self.bias_prior.sample(rng)

    @property
    @abc.abstractmethod
    def bias_prior(self):
        """Prior for the bias parameters."""

    @abc.abstractmethod
    def log_likelihood(self, data, reward_model, bias_params):
        """Compute log likelihood of given human data under the current reward
        and bias model parameters."""

    @abc.abstractmethod
    def log_likelihood_grad_rew(self, data, reward_model, bias_params):
        """Compute gradient of log likelihood of human data with respect to
        reward parameters only."""

    @abc.abstractmethod
    def log_likelihood_grad_bias(self, data, reward_model, bias_params):
        """Compute gradient of log likelihood of human data with respect to
        bias parameters only."""
