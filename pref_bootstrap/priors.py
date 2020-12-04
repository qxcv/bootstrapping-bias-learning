"""Priors for reward and bias models."""

import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
import numpy as np


class Prior(abc.ABC):
    """Base class for priors on reward parameters and bias model parameters."""
    @abc.abstractmethod
    def log_prior(self, weights):
        """Compute log prior for given weight values."""

    @abc.abstractmethod
    def log_prior_grad(self, weights):
        """Compute gradient of log prior with respect to weights."""

    def in_support(self, weights):
        """Are the given weights in the support of this prior?"""
        proj = self.project_to_support(self, weights)
        return np.allclose(proj, weights)

    @abc.abstractmethod
    def project_to_support(self, weights):
        """Project weights onto the support of this prior."""

    @abc.abstractmethod
    def sample(self, key):
        """Sample a vector of elements of the right shape."""


class FixedGaussianPrior(Prior):
    """Gaussian prior on parameters."""
    def __init__(self, shape, *, mean=0.0, std=1.0):
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0
        self.mean = jnp.float64(mean)
        self.std = jnp.float64(std)
        self.shape = shape

    def log_prior(self, params):
        """Return log likelihood of parameters under prior."""
        # log likelihood function, see:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        var = self.std**2
        nitems = params.size
        diff = params - self.mean
        scaled_sq_err_term = -0.5 * jnp.dot(diff, diff) / var
        # log determinant of covariance matrix
        log_det_cov_term = -nitems * jnp.log(self.std)
        norm_term = -0.5 * nitems * jnp.log(2 * jnp.pi)
        return log_det_cov_term + scaled_sq_err_term + norm_term

    def log_prior_grad(self, params):
        # gradient of above function w.r.t. params
        variance = self.std**2
        mean_diff = params - self.mean
        return -mean_diff / variance

    def in_support(self, weights):
        return jnp.all(jnp.isfinite(weights)) and jnp.all(jnp.isreal(weights))

    def project_to_support(self, weights):
        return weights

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key):
        key, inner_key = jrandom.split(key)
        vec = jrandom.normal(inner_key, shape=self.shape) * self.std + self.mean
        return key, vec


class ExponentialPrior(Prior):
    """Exponential prior, for positive parameters.

    (for reference: exponential density is lam*exp(-lam*x)), for scalar x.)"""
    def __init__(self, shape, *, lam=1.0):
        assert isinstance(lam, float)
        assert lam > 0
        self.shape = shape
        self.lam = lam

    def log_prior(self, params):
        nelem = params.size
        return -self.lam * jnp.sum(params) + nelem * jnp.log(self.lam)

    def log_prior_grad(self, params):
        return -self.lam * jnp.ones_like(params)

    def in_support(self, weights):
        return jnp.all(weights >= 0)

    def project_to_support(self, weights):
        return jnp.maximum(weights, 0.0)

    def sample(self, key):
        key, inner_key = jrandom.split(key)
        vec = jrandom.exponential(inner_key, shape=self.shape) / self.lam
        return key, vec


class BetaPrior(Prior):
    """Beta prior. (density is x^(alpha-1) * (1-x)^(beta-1)) / B(alpha,beta),
    where B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)."""
    def __init__(self, shape, *, alpha=0.5, beta=0.5, eps=1e-5):
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert alpha > 0
        assert beta > 0
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.shape = shape

    def log_prior(self, params):
        nelem = params.size
        gamma_term = nelem * (jsp.special.gammaln(self.alpha) +
                              jsp.special.gammaln(self.beta) -
                              jsp.special.gammaln(self.alpha + self.beta))
        alpha_term = (self.alpha - 1) * jnp.sum(jnp.log(params))
        beta_term = (self.beta - 1) * jnp.sum(jnp.log(1 - params))
        return alpha_term + beta_term - gamma_term

    def log_prior_grad(self, params):
        return (self.alpha - 1) / params - (self.beta - 1) / (1 - params)

    def in_support(self, weights):
        return jnp.all((weights >= self.eps) & (weights <= 1.0 - self.eps))

    def project_to_support(self, weights):
        return jnp.clip(weights, self.eps, 1.0 - self.eps)

    def sample(self, key):
        key, inner_key = jrandom.split(key)
        vec = jrandom.beta(key=key, a=self.alpha, b=self.beta, shape=self.shape)
        return key, vec
