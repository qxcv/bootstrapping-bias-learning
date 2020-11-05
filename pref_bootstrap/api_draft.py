"""Draft API for different feedback models."""

from collections import abc


class EnvFeedbackModel(abc.ABC):
    """Abstract base class for a 'feedback model' in some environment.

    A feedback model is implicitly associated with a specific feedback modality
    (e.g. demonstrations, corrections, paired comparisons), and explicitly
    associated with a specific environment. `EnvFeedbackModel` instances do not
    have state of their own (except environment-specific metadata), but they
    are able to generate reward and bias parameter vectors, as well as priors
    for those vectors. Later on, the generated reward and bias parameter
    vectors can be used to compute the log likelihood of some observed data,
    and to compute gradients of the log likelihood with respect to both reward
    parameters and bias parameters.

    PROBLEMS:

    - It feels weird that `create_{rew,bias}_prior` returns a distribution
      (which can evaluate log likelihoods, and potentially has its own
      parameters) while `init_{rew,bias}_params` just returns a flat parameter
      vector. If you want to do anything with the reward/bias parameter vector,
      you have to call methods of `EnvFeedbackModel`.

      This is probably going to lead to weird stuff if I consider more complex
      reward/bias models (e.g. do I have to store the NN architecture for the
      reward function as part of the `EnvFeedbackModel` so that it knows what
      to do with reward parameters?). Possibly there's a layer of abstraction
      missing, or an inappropriate abstraction somewhere.

    - This design doesn't account for algorithms that iteratively collect more
      data. Pretty much everything except IRL is likely to fall in this bucket
      (e.g. paired comparisons, corrections, scalar feedback, anything
      DAgger-like, etc.). Probably the `EnvFeedbackModel` also needs an
      _interaction model_ that makes it possible to get data. I don't know what
      shape that should have, though, or how it should fit into the outer loop.
    """
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def init_rew_params(self):
        """Generate a random set of reward model parameters for this
        feedback modality in this environment."""
        pass

    @abc.abstractmethod
    def create_rew_prior(self):
        """Create a probability distribution representing a prior for the
        reward parameters."""
        pass

    @abc.abstractmethod
    def init_bias_params(self):
        """Generate a random set of bias model parameters for this feedback
        modality in this environment."""
        pass

    @abc.abstractmethod
    def create_bias_prior(self):
        """Similar to `create_rew_prior`, this method generates a probability
        distribution representing a prior over bias parameters."""
        pass

    @abc.abstractmethod
    def log_likelihood(self, data, reward_params, bias_params):
        """Compute log likelihood of given human data under the current reward
        and bias model parameters."""
        pass

    @abc.abstractmethod
    def log_likelihood_grad_rew(self, data, reward_params, bias_params):
        """Compute gradient of log likelihood of human data with respect to
        reward parameters only."""
        pass

    @abc.abstractmethod
    def log_likelihood_grad_bias(self, data, reward_params, bias_params):
        """Compute gradient of log likelihood of human data with respect to
        bias parameters only."""
        pass
