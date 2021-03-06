"""Priors for reward and bias models."""

import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
import numpy as np
from jax import grad

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

    
class LogNormalPrior(Prior): 
    def __init__(self, shape, mean, std): 
        
        self.mu = mean
        self.shape = shape
        self.std = std
        
    def log_prior(self, params): 
        return -jnp.log(params - self.mu)**2/(2*self.std**2) - params*self.std*jnp.sqrt(2*jnp.pi)
    
    def log_prior_grad(self, params):
        return grad(self.log_prior)(params)
    
    def in_support(self, weights):
        return jnp.all(weights >= 0)

    def project_to_support(self, weights):
        return jnp.maximum(weights, 0.0)
    
    def sample(self, key):
        key, inner_key = jrandom.split(key)
        vec = jnp.exp(jrandom.normal(key, shape=self.shape)*self.std + self.mu)
        
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

    
class MixedPrior(Prior): 
    """
    For Sean feedback learner. 
    
    The first parameter is the temperature. This has an exp. prior
    The second param is the cutoff, which has a gaussian prior.
    
    """
    def __init__(self, shape1, mean1, std1 , mean2, std2): 
        self.mean1 = jnp.float64(mean1)
        self.std1 = jnp.float64(std1)
        self.shape1 = int(shape1)
        self.mean2 = jnp.float64(mean2)
        self.std2 = jnp.float64(std2)
        self.shape = 1
        self.p1 = FixedGaussianPrior((self.shape1,), mean=mean1, std=std1)
        self.p2 = FixedGaussianPrior((self.shape,), mean=mean2, std=std2)
        
        
    def log_prior(self, params):
        return self.p1.log_prior(params) + self.p2.log_prior(params)
    
    def log_prior_grad(self, params): 
        return (jnp.concatenate([(self.p1.log_prior_grad(params[:-1])), 
                               jnp.expand_dims(self.p2.log_prior_grad(params[-1]), axis=0)]))
    
    def in_support(self, params): 
        return jnp.concatenate([jnp.expand_dims(self.p1.in_support(params[:-1])), 
                               jnp.expand_dims(self.p2.in_support(params[-1]), axis=0)])
    
    def project_to_support(self, params):
        return jnp.concatenate([(self.p1.project_to_support(params[:-1])), 
                               jnp.expand_dims(self.p2.project_to_support(params[-1]), axis=0)])
    
    def sample(self, key): 
        k, v1 =self.p1.sample(key)
        k, v2 = self.p2.sample(k)
        vec = jnp.concatenate([
            v1, v2
        ])
        print('sampledvec', vec)
        return k, vec
    