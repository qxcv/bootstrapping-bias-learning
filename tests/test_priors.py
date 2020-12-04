import jax.random as jrandom
import jax.numpy as jnp
import pytest
from scipy import stats

import pref_bootstrap.priors as p
from .test_support import grad_check


def _gen_samples(key, dist_sample, n_samples):
    # TODO(sam): JIT this whole thing somehow (it's very slow)
    samples = []
    for s in range(n_samples):
        key, sample = dist_sample(key)
        samples.extend(sample)
    samples = jnp.stack(samples, axis=0).astype('float64')
    samples = samples.flatten()
    sample_mean = jnp.mean(samples, axis=0)
    sample_var = jnp.sum(
        (samples - sample_mean[None])**2, axis=0) / (samples.size - 1)
    sample_std = jnp.sqrt(sample_var)
    return key, sample_mean, sample_std


def _check_sample_stats(key, dist, expect_mean, expect_std, n_samples=10):
    key, sample_mean, sample_std = _gen_samples(key, dist.sample, n_samples)
    assert jnp.allclose(sample_mean, expect_mean, rtol=1e-2, atol=1e-3)
    assert jnp.allclose(sample_std, expect_std, rtol=1e-2, atol=1e-3)
    return key


GAUSS_PARAMS = [
    (0.0, 1.0),
    (-5.0, 1e-2),
]
GAUSS_VECS = [
    jnp.array([0.00359929, -0.00547784, -0.01242186]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([-3.59463682, -5.74071806, -5.16674389, -2.19397403]),
]


@pytest.mark.parametrize("mean,std", GAUSS_PARAMS)
@pytest.mark.parametrize("vec", GAUSS_VECS)
def test_gaussian_ll(mean, std, vec):
    prior = p.FixedGaussianPrior(shape=vec.shape, mean=mean, std=std)

    # check support
    assert prior.in_support(vec)
    assert jnp.allclose(prior.project_to_support(vec), vec)

    # check log prior
    our_log_prior = prior.log_prior(vec)
    ref_log_prior = jnp.sum(stats.norm.logpdf(vec, loc=mean, scale=std))
    assert jnp.allclose(our_log_prior, ref_log_prior)

    # check gradient
    grad_check(prior.log_prior, prior.log_prior_grad, vec, eps=1e-5)


@pytest.mark.parametrize("mean,std", GAUSS_PARAMS)
def test_gaussian_sample(mean, std):
    prior = p.FixedGaussianPrior(shape=(300, 300), mean=mean, std=std)
    key = jrandom.PRNGKey(42)
    _check_sample_stats(key=key, dist=prior, expect_mean=mean, expect_std=std)


EXPONENTIAL_PARAMS = [
    1.0,
    3e-2,
]
EXPONENTIAL_TEST_VECS = [
    jnp.array([1.4]),
    jnp.array([1e-2, 8e-3, 7e-2]),
    jnp.array([0.0, 92.5]),
]


@pytest.mark.parametrize("lam", EXPONENTIAL_PARAMS)
@pytest.mark.parametrize("vec", EXPONENTIAL_TEST_VECS)
def test_exponential_ll(lam, vec):
    prior = p.ExponentialPrior(shape=vec.shape, lam=lam)

    # check support
    assert prior.in_support(vec)
    assert jnp.allclose(prior.project_to_support(vec), vec)
    unsupported = jnp.array([1.0, -1.0, 4.2])
    assert not prior.in_support(unsupported)
    proj_unsupported = prior.project_to_support(unsupported)
    assert prior.in_support(proj_unsupported)

    # check log pior
    our_log_prior = prior.log_prior(vec)
    # scipy.stats has a weird exponential dist parameterisation; see
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    ref_log_prior = jnp.sum(stats.expon.logpdf(vec, scale=1 / lam))
    assert jnp.allclose(our_log_prior, ref_log_prior)

    grad_check(prior.log_prior, prior.log_prior_grad, vec, eps=1e-5)


@pytest.mark.parametrize("lam", EXPONENTIAL_PARAMS)
def test_exponential_sample(lam):
    prior = p.ExponentialPrior(shape=(300, 300), lam=lam)
    key = jrandom.PRNGKey(42)
    _check_sample_stats(
        key=key, dist=prior, expect_mean=1.0 / lam, expect_std=1.0 / lam)


BETA_PARAMS = [
    (0.5, 0.5),
    (1.0, 2.0),
    (0.3, 4.0),
]
BETA_TEST_VECS = [
    jnp.array([0.5]),
    jnp.array([1-5e-3, 0.9, 0.98]),
    jnp.array([1e-2, 0.2])
]


@pytest.mark.parametrize("alpha,beta", BETA_PARAMS)
@pytest.mark.parametrize("vec", BETA_TEST_VECS)
def test_beta_ll(alpha, beta, vec):
    prior = p.BetaPrior(shape=vec.shape, alpha=alpha, beta=beta)

    # check support (including projection, which is weird for beta since it
    # potentially has an open support)
    assert prior.in_support(vec)
    assert jnp.allclose(prior.project_to_support(vec), vec)
    unsupported = jnp.array([-0.1, 0.3, 1.4])
    assert not prior.in_support(unsupported)
    proj_unsupported = prior.project_to_support(unsupported)
    assert prior.in_support(proj_unsupported)

    # check value
    our_log_prior = prior.log_prior(vec)
    ref_log_prior = jnp.sum(stats.beta.logpdf(vec, alpha, beta))
    assert jnp.allclose(our_log_prior, ref_log_prior)

    # check grad
    grad_check(prior.log_prior, prior.log_prior_grad, vec, eps=1e-5)


@pytest.mark.parametrize("alpha,beta", BETA_PARAMS)
def test_beta_sample(alpha, beta):
    prior = p.BetaPrior(shape=(300, 300), alpha=alpha, beta=beta)
    key = jrandom.PRNGKey(42)
    expect_std = jnp.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
    _check_sample_stats(
        key=key, dist=prior, expect_mean=alpha/(alpha+beta), expect_std=expect_std)
