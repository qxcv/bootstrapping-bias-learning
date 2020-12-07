"""Abstract base class for experts/demonstrators."""
import abc

import numpy as np

from pref_bootstrap.algos.mce_irl import mce_irl_sample


class Expert(abc.ABC):
    """Abstract base class for experts"""

    def __init__(self, env, seed):
        self.env = env
        assert isinstance(seed, int)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    @abc.abstractmethod
    def interact(self, *queries):
        """Respond to an environment query. The form of query and the expected
        response will vary depending on the algorithm."""
        pass


class PairedComparisonExpert(Expert):
    """Boltzmann-rational paired comparison expert."""

    def __init__(self, *args, boltz_temp=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert boltz_temp is None or boltz_temp > 0
        self.boltz_temp = boltz_temp

    def interact(self, qtraj1, qtraj2):
        """Do a Boltzmann comparison of two trajectories based on reward.
        Return True if this randomised comparison claims qtraj1 > qtraj2 (i.e.
        first trajectory is preferable to second)."""
        assert isinstance(qtraj1, dict) and "states" in qtraj1
        assert isinstance(qtraj2, dict) and "states" in qtraj2
        reward_mat = self.env.reward_matrix
        rew1 = np.sum(reward_mat[qtraj1["states"]])
        rew2 = np.sum(reward_mat[qtraj2["states"]])
        exp_rew1 = np.exp(self.boltz_temp * rew1)
        exp_rew2 = np.exp(self.boltz_temp * rew2)
        p_traj1 = exp_rew1 / (exp_rew1 + exp_rew2)
        return self.rng.rand() < p_traj1


class MEDemonstratorExpert(Expert):
    def interact(self, n_demos):
        return mce_irl_sample(self.env, n_demos)

class ScalarFeedbackExpert(Expert):
    """Gaussisan-corrupted function"""
    def __init__(self, *args, gauss_mean= 0.0, gauss_std=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
    
    def interact(self, traj):
        """Corrupt each observation with Gaussian Noise"""
        assert isinstance(traj, dict) and 'states' in traj
        reward_mat = self.env.reward_matrix
        # rewards = np.sum(reward_mat[traj['states']], axis=1)
        rewards = reward_mat[traj['states']]
        # print(rewards.shape)
        # rewards = rewards.reshape((rewards.shape[0], 1))
        norm_rew = np.random.normal(loc=self.gauss_mean, scale=self.gauss_std, size=rewards.shape)
        scalar_feedback = rewards + norm_rew
        return {
        'trajectories': traj,
        'corrupted_rewards': scalar_feedback,
        }
