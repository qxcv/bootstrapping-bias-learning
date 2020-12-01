"""Abstract base class for experts/demonstrators."""
import abc

import numpy as np

from pref_bootstrap.algos.mce_irl import mce_irl_sample
import random

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
    
    
class TopKExpert(Expert): 
    def __init__(self, *args, temp=.3, K=.2, **kwargs):
        """
        inputs
        temp: How big of an uncertainty do we have (smaller = more uncertainty)
        K: What percentage of trials will we label as having being optimal? 
        """
        
        
        # Here we take in the K as a percentage.
        super().__init__(*args, **kwargs)
        self.temp=temp # how uncertain we are.
        assert K > 0.0 and K <= 1.0
        self.K = K
        self.cutoff = 0.0
        
    def interact(self, n_demos): 
        """
        input: n_demos, a bunch of demos. 
        output: Label in [1, 0] that each demo is in the top-K best demos. 
        """
        reward_mat = self.env.reward_matrix
        rews = [(np.sum(reward_mat[demo])) for demo in n_demos['states']]
        rews = np.array(rews)
        
        # determine the cutoff
        assert self.K <= 1.0
        cutoff = int(self.K*len(rews))
        
        
        rews_sorted = np.sort(rews)[::-1]
        self.cutoff = rews_sorted[cutoff] # This is our decision boundary on wether or not 
                                  # to include the data or not. 
        print('cutoff', self.cutoff)
        labels = np.array([self.label(y) for y in rews])
        return labels
        
    def label(self, x):
        y = self.temp*(x-self.cutoff)
        sig = 1/(1+np.exp(-y))
        p_topk = sig
        return p_topk > random.random()
        
        
        
