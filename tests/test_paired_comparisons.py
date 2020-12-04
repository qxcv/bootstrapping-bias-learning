"""Test log likelihood and gradient for paired comparisons."""
import numpy as np
import pytest
import jax.random as jrandom

from pref_bootstrap.algos import mce_irl
from pref_bootstrap.envs import gridworld, mdp_interface
import pref_bootstrap.expert_base as experts
import pref_bootstrap.feedback_learner_paired_comparisons as fbl_cmp
import pref_bootstrap.reward_models as rmodels
from .test_support import grad_check

SEEDS = [7, 13, 42]


def generate_comparison_dataset(env, pc_ntraj, seed):
    # FIXME(sam): this is copy-pasted from the notebook too; factor it out!
    pc_expert = experts.PairedComparisonExpert(env, boltz_temp=1.0, seed=42)
    pc_trajectories = mce_irl.mce_irl_sample(
        env, pc_ntraj, R=np.ones((env.n_states, )))
    comparisons = []
    for first_idx in range(pc_ntraj):
        second_idx = np.random.randint(pc_ntraj - 1)
        if second_idx >= first_idx:
            second_idx += 1
        traj1_is_better = pc_expert.interact(
            dict(states=pc_trajectories['states'][first_idx]),
            dict(states=pc_trajectories['states'][second_idx]))
        if traj1_is_better:
            # the better trajectory comes before the worse one
            comparisons.append((first_idx, second_idx))
        else:
            comparisons.append((second_idx, first_idx))
    return {
        'trajectories': pc_trajectories,
        'comparisons': np.asarray(comparisons),
    }


@pytest.mark.parametrize('seed', SEEDS)
def test_fbl_cmp_ll_grad(seed):
    """Make sure that log likelihood and gradients agree with each other."""
    key = jrandom.PRNGKey(seed)

    # generate some seeds for environment and reward model
    key, rmodel_key, env_key, ds_key = jrandom.split(key, num=4)

    def gen_randint(key):
        return int(jrandom.randint(key, shape=(), minval=0, maxval=2**31 - 1))

    rmodel_seed = gen_randint(rmodel_key)
    env_seed = gen_randint(env_key)
    ds_seed = gen_randint(ds_key)
    del rmodel_key, env_key, ds_key

    # create the environment
    random_gridworld = gridworld.GridworldMdp.generate_random(
        4, 4, 0.2, 0.1, seed=env_seed)
    env = mdp_interface.GridworldEnvWrapper(
        random_gridworld, random_gridworld.height + random_gridworld.width)

    # create the feedback model, reward model, etc.
    fbm = fbl_cmp.PairedCompFeedbackModel(env)
    key, bias_params = fbm.bias_prior.sample(key)
    rmodel = rmodels.LinearRewardModel(env.obs_dim, seed=rmodel_seed)
    rmodel_params = rmodel.get_params()

    # create some fake data
    dataset = generate_comparison_dataset(env, 10, ds_seed)

    # convenience functions
    def rmodel_from_params(params):
        model = rmodels.LinearRewardModel(env.obs_dim, seed=0)
        model.set_params(params)
        return model

    def log_likelihood(rparams=rmodel_params, bparams=bias_params):
        inner_rmodel = rmodel_from_params(rparams)
        return fbm.log_likelihood(dataset, inner_rmodel, bparams)

    def log_likelihood_grad_wrt_r(rparams):
        inner_rmodel = rmodel_from_params(rparams)
        return fbm.log_likelihood_grad_rew(dataset, inner_rmodel, bias_params)

    def log_likelihood_grad_wrt_b(bparams):
        return fbm.log_likelihood_grad_bias(dataset, rmodel, bparams)

    # finally, test that the log likelihood and gradient match up for both bias
    # and reward
    grad_check(
        lambda r: log_likelihood(rparams=r),
        log_likelihood_grad_wrt_r,
        rmodel_params)
    grad_check(
        lambda b: log_likelihood(bparams=b),
        log_likelihood_grad_wrt_b,
        bias_params)
