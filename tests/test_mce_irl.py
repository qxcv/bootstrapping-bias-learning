"""Test log likelihood and gradient for MCE IRL."""
import pytest
import jax.random as jrandom

from pref_bootstrap.envs import gridworld, mdp_interface
import pref_bootstrap.expert_base as experts
import pref_bootstrap.feedback_learner_blind_irl as fbl_blind_irl
import pref_bootstrap.reward_models as rmodels
from .test_support import grad_check

SEEDS = [7, 13, 42]


@pytest.mark.parametrize('seed', SEEDS)
def test_fbl_cmp_ll_grad(seed):
    """Ensure that log likelihood & gradient are consistent for MCE IRL."""
    key = jrandom.PRNGKey(seed)

    # generate some seeds for environment and reward model
    key, inner_key = jrandom.split(key)
    rmodel_seed, env_seed, ds_seed = map(
        int,
        jrandom.randint(inner_key, shape=(3, ), minval=0, maxval=2**31 - 1))
    del inner_key

    # create the environment
    random_gridworld = gridworld.GridworldMdp.generate_random(
        4, 4, 0.2, 0.1, seed=env_seed)
    env = mdp_interface.GridworldEnvWrapper(
        random_gridworld, random_gridworld.height + random_gridworld.width)

    # create the feedback model, reward model, etc.
    fbm_irl = fbl_blind_irl.BlindIRLFeedbackModel(env)
    key, bias_params = fbm_irl.bias_prior.sample(key)
    rmodel = rmodels.LinearRewardModel(env.obs_dim, seed=rmodel_seed)
    rmodel_params = rmodel.get_params()

    # create some fake data
    expert = experts.MEDemonstratorExpert(env, ds_seed)
    dataset = expert.interact(10)

    # convenience functions
    def rmodel_from_params(params):
        model = rmodels.LinearRewardModel(env.obs_dim, seed=0)
        model.set_params(params)
        return model

    def log_likelihood(rparams=rmodel_params, bparams=bias_params):
        inner_rmodel = rmodel_from_params(rparams)
        return fbm_irl.log_likelihood(dataset, inner_rmodel, bparams)

    def log_likelihood_grad_wrt_r(rparams):
        inner_rmodel = rmodel_from_params(rparams)
        return fbm_irl.log_likelihood_grad_rew(dataset, inner_rmodel,
                                               bias_params)

    def log_likelihood_grad_wrt_b(bparams):
        return fbm_irl.log_likelihood_grad_bias(dataset, rmodel, bparams)

    grad_check(
        lambda r: log_likelihood(rparams=r),
        log_likelihood_grad_wrt_r,
        rmodel_params)
    grad_check(
        lambda b: log_likelihood(bparams=b),
        log_likelihood_grad_wrt_b,
        bias_params)
