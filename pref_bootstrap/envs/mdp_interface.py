import abc
import random

import gym
from gym import spaces
import numpy as np


class Mdp(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """
    # FIXME(sam): replace this with the ModelBaseEnv abstraction from
    # `imitation` (or just a hand-rolled abstraction).

    def __init__(self, mdp):
        self.gridworld = mdp
        self.reset()

    def get_current_state(self):
        return self.state

    def get_actions(self, state):
        return self.gridworld.get_actions(state)

    def perform_action(self, action):
        """Performs the action, updating the state and providing a reward."""
        state = self.get_current_state()
        next_state, reward = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action):
        """Chooses the next state according to T(state, action)."""
        rand = random.random()
        sum = 0.0
        results = self.gridworld.get_transition_states_and_probs(state, action)
        for next_state, prob in results:
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                reward = self.gridworld.get_reward(state, action)
                return (next_state, reward)
        raise ValueError('Total transition probability less than one.')

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.gridworld.get_start_state()

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.gridworld.is_terminal(self.get_current_state())


class ModelBasedEnv(gym.Env, abc.ABC):
    """ABC for small, discrete, finite-horizon MDPs."""

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @property
    @abc.abstractmethod
    def transition_matrix(self):
        """3D transition matrix.
        Dimensions correspond to current state, current action, and next state.
        In other words, if `T` is our returned matrix, then `T[s,a,sprime]` is
        the chance of transitioning into state `sprime` after taking action `a`
        in state `s`.
        """

    @property
    @abc.abstractmethod
    def observation_matrix(self):
        """2D observation matrix.
        Dimensions correspond to current state (first dim) and elements of
        observation (second dim)."""

    @property
    @abc.abstractmethod
    def reward_matrix(self):
        """1D reward matrix with an element corresponding to each state."""

    @property
    @abc.abstractmethod
    def horizon(self):
        """Number of actions that can be taken in an episode."""

    @property
    @abc.abstractmethod
    def initial_state_dist(self):
        """1D vector representing a distribution over initial states."""
        return

    # ############################### #
    # SUPERCLASS-SUPPLIED METHODS     #
    # ############################### #

    def __init__(self):
        self._state_space = None
        self._observation_space = None
        self._action_space = None
        self.cur_state = None
        self._n_actions_taken = None
        self.seed()

    @property
    def state_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in
        # subclasses.
        if self._state_space is None:
            self._state_space = spaces.Discrete(self.state_dim)
        return self._state_space

    @property
    def observation_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in
        # subclasses.
        if self._observation_space is None:
            self._observation_space = spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(self.obs_dim,)
            )
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in
        # subclasses.
        if self._action_space is None:
            self._action_space = spaces.Discrete(self.n_actions)
        return self._action_space

    @property
    def n_actions_taken(self) -> int:
        """Number of steps taken so far."""
        return self._n_actions_taken

    def seed(self, seed=None):
        if seed is None:
            # Gym API wants list of seeds to be returned for some reason, so
            # generate a seed explicitly in this case
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.cur_state = self.initial_state()
        self._n_actions_taken = 0
        return self.obs_from_state(self.cur_state)

    def step(self, action):
        if self.cur_state is None or self._n_actions_taken is None:
            raise ValueError("Need to call reset() before first step()")

        old_state = self.cur_state
        self.cur_state = self.transition(self.cur_state, action)
        obs = self.obs_from_state(self.cur_state)
        rew = self.reward(old_state, action, self.cur_state)
        done = self.terminal(self.cur_state, self._n_actions_taken)
        self._n_actions_taken += 1

        infos = {"old_state": old_state, "new_state": self.cur_state}
        return obs, rew, done, infos

    def initial_state(self):
        return self.rand_state.choice(self.n_states, p=self.initial_state_dist)

    def transition(self, state, action):
        out_dist = self.transition_matrix[state, action]
        choice_states = np.arange(self.n_states)
        return int(self.rand_state.choice(choice_states, p=out_dist, size=()))

    def reward(self, state, action, new_state):
        reward = self.reward_matrix[state]
        assert np.isscalar(reward), reward
        return reward

    def terminal(self, state, n_actions_taken):
        return n_actions_taken >= self.horizon

    def obs_from_state(self, state):
        # Copy so it can't be mutated in-place (updates will be reflected in
        # self.observation_matrix!)
        obs = self.observation_matrix[state].copy()
        assert obs.ndim == 1, obs.shape
        return obs

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    @property
    def state_dim(self):
        """Size of state vectors for this MDP."""
        return self.observation_matrix.shape[0]

    @property
    def obs_dim(self):
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[1]


class GridworldEnvWrapper(ModelBasedEnv):
    """Wrap a gridworld as a ModelBasedEnv."""

    def __init__(self, gridworld, horizon):
        super().__init__()
        # Some useful gridworld attributes:
        # - .rewards is a dict mapping (x,y) locations to reward values. Not
        #   all locations have a reward, and there is a "living reward" that
        #   you need to remember to add if the agent is not staying in place
        #   (we will probably omit that).
        # - .start_state is the initial state (an (x, y) location)
        self.gridworld = gridworld

        n_states = self.gridworld.height * self.gridworld.width

        # populate transitions
        self._trans_mat = self.gridworld.get_transition_matrix()

        # observations are just indicators telling you which state you're in
        self._obs_mat = np.eye(self.n_states)

        # populate rewards
        self._rew_mat = np.zeros((n_states, ))
        for xy_state, rew in self.gridworld.rewards.items():
            int_state = self._convert_xy_state(xy_state)
            self._rew_mat[int_state] = rew

        # horizon is supplied by user
        # TODO(sam): if horizon is not supplied, infer from max of APSP on the
        # gridworld
        self._horizon = horizon

        # initial state distribution is deterministic
        self._init_state_dist = np.zeros((n_states, ))
        int_init_state = self._convert_xy_state(self.gridworld.start_state)
        self._init_state_dist[int_init_state] = 1.0

    def _convert_xy_state(self, xy_state):
        """Convert an (x,y) location (used as the 'state' in GridworldMdpNoR
        and friends) into an integer representing state number (used in this
        class)."""
        x, y = xy_state
        return y * self.gridworld.width + x

    @property
    def transition_matrix(self):
        """3D transition matrix. (shape [|S|,|A|,|S|], final axis represents
        destination state & slicing over final axis thus yields normalised
        categorical distribution)."""
        return self._trans_mat

    @property
    def observation_matrix(self):
        """2D observation matrix (shape [|S|,d])."""
        return self._obs_mat

    @property
    def reward_matrix(self):
        """1D reward matrix (shape [|S|])."""
        return self._rew_mat

    @property
    def horizon(self):
        """Horizon (int)."""
        return self._horizon

    @property
    def initial_state_dist(self):
        """Initial state distribution (shape [|S|])."""
        return self._init_state_dist
