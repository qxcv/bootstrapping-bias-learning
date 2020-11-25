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
