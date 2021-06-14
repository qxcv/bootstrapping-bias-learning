from functools import lru_cache

import numpy as np

from pref_bootstrap.envs.mdp_interface import ModelBasedEnv


class VaseWorld(ModelBasedEnv):
    """Simple gridworld that requires a ~moderately complex policy to solve.
    Agent must reach goal while collecting diamonds & avoiding vases/lava. Also
    has a slippery location where the agent could transition into any adjacent
    square, not just the desired one."""
    _horizon = 15
    reward_weights = np.array([
        # goal (0)
        20.0,
        # lava (1)
        -160.0,
        # diamond (2)
        4.0,
        # vase (3)
        -4.0,
        # slip (4)
        0.0,
        # distractor (5)
        0.0,
    ], dtype='float32')
    slip_plane = np.array([
        # is this a "slippery" square?
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype='float32')
    feature_planes = np.array([
        [
            # goal (0)
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            # lava (1)
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        [
            # diamond (2)
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            # vase (3)
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        # slip (4)
        slip_plane.tolist(),
        [
            # distractor (5)
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
        ],
    ], dtype='float32')
    n_rows, n_cols = feature_planes.shape[1:]

    @property
    @lru_cache(maxsize=None)
    def observation_matrix(self) -> np.ndarray:
        feat_planes_trans = np.transpose(self.feature_planes, (1, 2, 0))
        result = feat_planes_trans.reshape((-1, feat_planes_trans.shape[-1]))
        assert result.shape == (self.n_states, self.feature_planes.shape[0])
        return result

    @property
    @lru_cache(maxsize=None)
    def transition_matrix(self) -> np.ndarray:
        # left/right/up/down actions (tuples indicate action number & effect
        # the action has on the row/col)
        actions = [(0, (0, -1)), (1, (0, +1)), (2, (+1, 0)), (3, (-1, 0))]

        # initialize the transition matrix as all zeros
        n_states = self.n_rows * self.n_cols
        n_acts = len(actions)
        trans_matrix = np.zeros((n_states, n_acts, n_states))

        # dims of transition matrix are s,a,s', so we build one action/row/col
        # at a time
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                src_state_num = col + row * self.n_cols
                for act_num, act_delta in actions:
                    tgt_row = np.clip(row + act_delta[0], 0, self.n_rows - 1)
                    tgt_col = np.clip(col + act_delta[1], 0, self.n_cols - 1)
                    tgt_state_num = tgt_col + tgt_row * self.n_cols
                    if self.slip_plane[row, col]:
                        # with probability p_slip, replace the chosen action
                        # with a uniform random action
                        p_slip = 0.8
                        trans_matrix[src_state_num, act_num, tgt_state_num] \
                            = 1 - p_slip
                        for _, alt_delta in actions:
                            slip_row = np.clip(row + alt_delta[0], 0, self.n_rows - 1)
                            slip_col = np.clip(col + alt_delta[1], 0, self.n_cols - 1)
                            slip_state_num = slip_col + slip_row * self.n_cols
                            trans_matrix[src_state_num, act_num, slip_state_num] \
                                += p_slip / len(actions)
                    else:
                        # otherwise, transition as usual
                        trans_matrix[src_state_num, act_num, tgt_state_num] = 1.0
                    tm_slice = trans_matrix[src_state_num, act_num, :]
                    assert np.allclose(np.sum(tm_slice), 1.0), tm_slice

        # make sure transition matrix is normalized
        assert np.all(trans_matrix >= 0.0), trans_matrix
        assert np.allclose(np.sum(trans_matrix, axis=2), 1.0), trans_matrix

        return trans_matrix

    @property
    @lru_cache(maxsize=None)
    def reward_matrix(self) -> np.ndarray:
        reward_vec = self.observation_matrix @ self.reward_weights
        assert reward_vec.shape == (self.n_states, ), reward_vec
        return reward_vec

    @property
    @lru_cache(maxsize=None)
    def horizon(self) -> int:
        # may need to tune this
        return self._horizon

    @property
    @lru_cache(maxsize=None)
    def initial_state_dist(self) -> np.ndarray:
        """1D vector representing a distribution over initial states."""
        init_dist = np.zeros((self.n_states, ))
        # low left hand corner
        n_rows, n_cols = self.feature_planes.shape[1:]
        init_state = (n_rows - 1) * n_cols
        init_dist[init_state] = 1.0
        assert np.allclose(np.sum(init_dist), 1.0), init_dist
        assert np.all(init_dist >= 0.0), init_dist
        return init_dist
