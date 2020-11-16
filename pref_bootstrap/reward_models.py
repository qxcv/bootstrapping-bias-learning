import abc

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.experimental.stax as jstax
import numpy as np


class RewardPrior(abc.ABC):
    """Abstract base class for priors on reward function parameters."""
    @abc.abstractmethod
    def log_prior(self, params):
        r"""Compute $\log p({\rm params})$ under this prior."""

    @abc.abstractmethod
    def log_prior_grad(self, inputs):
        r"""Compute \nabla_{\rm params} \log p({\rm params})."""

    @abc.abstractmethod
    def set_hyperparams(self, params):
        """Set a new hyperparameter vector for the prior (from flat Numpy
        array)."""

    @abc.abstractmethod
    def get_hyperparams(self):
        """Get current hyperparameter vector from model (as flat Numpy
        array)."""


class FixedGaussianRewardPrior(abc.ABC):
    """Gaussian prior on reward function parameters."""
    def __init__(self, *, mean=0.0, std=1.0):
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0
        self.mean = mean
        self.std = std

    def log_prior(self, params):
        """Return log likelihood of parameters under prior."""
        # log likelihood function, see:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        variance = self.std ** 2
        ndim = params.ndim
        mean_diff = params - self.mean
        scaled_sq_err = jnp.dot(mean_diff, mean_diff) / variance
        # log determinant of covariance matrix
        log_det_cov = 2 * ndim * jnp.log(self.std)
        norm_term = ndim * jnp.log(2 * jnp.pi)
        return -0.5 * (log_det_cov + scaled_sq_err + norm_term)

    def log_prior_grad(self, params):
        # gradient of above function w.r.t. params
        variance = self.std ** 2
        mean_diff = params - self.mean
        return -mean_diff / variance

    def set_hyperparams(self, params):
        raise NotImplementedError("This prior has no hyperparameters")

    def get_hyperparams(self):
        raise NotImplementedError("This prior has no hyperparameters")


class RewardModel(abc.ABC):
    """Abstract model for reward functions (linear, MLPs, nearest-neighbour,
    etc.)"""

    @abc.abstractmethod
    def out(self, inputs):
        """Get rewards for a batch of observations.
        Args:
            inputs (np.ndarray): 2D matrix of observations, with first axis
                most likely indexing over state & second indexing over elements
                of observations themselves.
        Returns:
            np.ndarray of rewards (just a 1D vector with one element for each
            supplied observation).
        """

    @abc.abstractmethod
    def grads(self, inputs):
        """Gradients of reward with respect to a batch of input observations.
        Args:
            inputs (np.ndarray): 2D matrix of observations, like .out().
        Returns:
            np.ndarray of gradients *with respect to each input separately*.
            e.g if the model has a W-dimensional parameter vector, and there
            are O observation passed in, then the return value will be an O*W
            matrix of gradients.
        """

    def out_grads(self, inputs):
        """Combination method to do forward-prop AND back-prop. This is trivial for
        linear models, but might provide some cost saving for deep ones.
        Args:
            inputs (np.ndarray): 2D matrix of observations, like .out().
        Returns:
            (np.ndarray, np.ndarray), where first array is equivalent to return
            value of .out() and second array is equivalent to return value of
            .grads().
        """
        return self.out(inputs), self.grads(inputs)

    @abc.abstractmethod
    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array).
        Args:
            params (np.ndarray): 1D parameter vector for the model.
        """

    @abc.abstractmethod
    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array).
        Args: empty.
        Returns:
            np.ndarray: 1D parameter vector for the model.
        """


class LinearRewardModel(RewardModel):
    """Linear reward model (without bias)."""

    def __init__(self, obs_dim, *, seed=None):
        """Construct linear reward model for `obs_dim`-dimensional observation space.
        Initial values are generated from given seed (int or None).
        Args:
            obs_dim (int): dimensionality of observation space.
            seed (int or None): random seed for generating initial params. If
                None, seed will be chosen arbitrarily
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self._weights = rng.randn(obs_dim)

    def out(self, inputs):
        in_shape = inputs.shape
        w_shape = self._weights.shape
        assert in_shape[-1:] == w_shape, (in_shape, w_shape)
        return inputs @ self._weights

    def grads(self, inputs):
        assert inputs.shape[1:] == self._weights.shape
        return inputs

    def set_params(self, params):
        assert params.shape == self._weights.shape
        self._weights = params

    def get_params(self):
        return self._weights


class JaxRewardModel(RewardModel, abc.ABC):
    """Wrapper for arbitrary Jax-based reward models.
    Useful for neural nets.
    """

    def __init__(self, obs_dim, *, seed=None):
        """Internal setup for Jax-based reward models.
        Initialises reward model using given seed & input size (`obs_dim`).
        Args:
            obs_dim (int): dimensionality of observation space.
            seed (int or None): random seed for generating initial params. If
                None, seed will be chosen arbitrarily, as in
                LinearRewardModel.
        """
        # TODO: apply jax.jit() to everything in sight
        net_init, self._net_apply = self.make_stax_model()
        if seed is None:
            # oh well
            seed = np.random.randint((1 << 63) - 1)
        rng = jrandom.PRNGKey(seed)
        out_shape, self._net_params = net_init(rng, (-1, obs_dim))
        self._net_grads = jax.grad(self._net_apply)
        # output shape should just be batch dim, nothing else
        assert out_shape == (-1, ), "got a weird output shape %s" % (
            out_shape, )

    @abc.abstractmethod
    def make_stax_model(self):
        """Build the stax model that this thing is meant to optimise.
        Should return (net_init, net_apply) pair, just like Stax modules.
        Returns:
            tuple of net_init(rng, input_shape) function to initialise the
            network, and net_apply(params, inputs) to do forward prop on the
            network.
        """

    def _flatten(self, matrix_tups):
        """Flatten everything and concatenate it together."""
        out_vecs = [v.flatten() for t in matrix_tups for v in t]
        return jnp.concatenate(out_vecs)

    def _flatten_batch(self, matrix_tups):
        """Flatten all except leading dim & concatenate results together in channel dim.
        (Channel dim is whatever the dim after the leading dim is)."""
        out_vecs = []
        for t in matrix_tups:
            for v in t:
                new_shape = (v.shape[0], )
                if len(v.shape) > 1:
                    new_shape = new_shape + (np.prod(v.shape[1:]), )
                out_vecs.append(v.reshape(new_shape))
        return jnp.concatenate(out_vecs, axis=1)

    def out(self, inputs):
        return np.asarray(self._net_apply(self._net_params, inputs))

    def grads(self, inputs):
        in_grad_partial = jax.partial(self._net_grads, self._net_params)
        grad_vmap = jax.vmap(in_grad_partial)
        rich_grads = grad_vmap(inputs)
        flat_grads = np.asarray(self._flatten_batch(rich_grads))
        assert flat_grads.ndim == 2 and flat_grads.shape[0] == inputs.shape[0]
        return flat_grads

    def set_params(self, params):
        # have to reconstitute appropriately-shaped weights from 1D param vec
        # shit this is going to be annoying
        idx_acc = 0
        new_params = []
        for t in self._net_params:
            new_t = []
            for v in t:
                new_idx_acc = idx_acc + v.size
                new_v = params[idx_acc:new_idx_acc].reshape(v.shape)
                # this seems to cast it to Jax DeviceArray appropriately;
                # surely there's better way, though?
                new_v = 0.0 * v + new_v
                new_t.append(new_v)
                idx_acc = new_idx_acc
            new_params.append(new_t)
        self._net_params = new_params

    def get_params(self):
        return self._flatten(self._net_params)


class MLPRewardModel(JaxRewardModel):
    """Simple MLP-based reward function with Jax/Stax."""

    def __init__(self, obs_dim, hiddens, activation="Tanh", **kwargs):
        """Construct an MLP-based reward function.
        Args:
            obs_dim (int): dimensionality of observation space.
            hiddens ([int]): size of hidden layers.
            activation (str): name of activation (Tanh, Relu, Softplus
                supported).
            **kwargs: extra keyword arguments to be passed to
                JaxRewardModel.__init__().
        """
        assert activation in ["Tanh", "Relu", "Softplus"], (
            "probably can't handle activation '%s'" % activation)
        self._hiddens = hiddens
        self._activation = activation
        super().__init__(obs_dim, **kwargs)

    def make_stax_model(self):
        act = getattr(jstax, self._activation)
        layers = []
        for h in self._hiddens:
            layers.extend([jstax.Dense(h), act])
        layers.extend([jstax.Dense(1), _StaxSqueeze()])
        return jstax.serial(*layers)


def _StaxSqueeze(axis=-1):
    """Stax layer that collapses a single axis that has dimension 1.
    Only used in MLPRewardModel.
    """

    def init_fun(rng, input_shape):
        ax = axis
        if ax < 0:
            ax = len(input_shape) + ax
        assert ax < len(
            input_shape), "invalid axis %d for %d-dimensional tensor" % (
                axis, len(input_shape), )
        assert input_shape[ax] == 1, "axis %d is %d, not 1" % (axis,
                                                               input_shape[ax])
        output_shape = input_shape[:ax] + input_shape[ax + 1:]
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.squeeze(inputs, axis=axis)

    return init_fun, apply_fun
