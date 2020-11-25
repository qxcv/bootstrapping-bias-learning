"""Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL), and
associated bias/rationality models. Copied from:

  https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/tabular_irl.py

(originally I (Sam) wrote that code for EE227C)

Follows the description in chapters 9 and 10 of Brian Ziebart's `PhD thesis`_.
Uses NumPy-based optimizer Jax, so the code can be run without
PyTorch/TensorFlow, and some code for simple reward models.

.. _PhD thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf
"""

import logging

import numpy as np
import scipy


def mce_partition_fh(env, *, R=None):
    r"""Performs the soft Bellman backup for a finite-horizon, undiscounted MDP.
    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).
    Args:
        env (ModelBasedEnv): a tabular, known-dynamics MDP.
        R (None or np.array): a reward matrix. Defaults to env.reward_matrix.
    Returns:
        (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        \pi is a 3d array, indexed \pi[t,s,a].
    """

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix

    # Initialization
    # indexed as V[t,s]
    V = np.full((horizon, n_states), -np.inf)
    # indexed as Q[t,s,a]
    Q = np.zeros((horizon, n_states, n_actions))
    broad_R = R[:, None]

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
        next_values_s_a = T @ V[t + 1, :]
        Q[t, :, :] = broad_R + next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def mce_occupancy_measures(env, *, R=None, pi=None):
    """Calculate state visitation frequency Ds for each state s under a given
    policy pi. You can get pi from `mce_partition_fh`.
    Args:
        env (ModelBasedEnv): a tabular MDP.
        R (None or np.ndarray): reward matrix. Defaults is env.reward_matrix.
        pi (None or np.ndarray): policy to simulate. Defaults to soft-optimal
            policy w.r.t reward matrix.
    Returns:
        Tuple of Dt (ndarray) and D (ndarray). D is an :math:`|S|`-dimensional
        vector recording the expected number of times each state is visited.
        Dt is a :math:`T*|S|`-dimensional vector recording the probability of
        being in a given state at a given timestep.
    """

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, R=R)

    D = np.zeros((horizon, n_states))
    D[0, :] = env.initial_state_dist
    for t in range(1, horizon):
        for a in range(n_actions):
            E = D[t - 1] * pi[t - 1, :, a]
            D[t, :] += E @ T[:, a, :]

    return D, D.sum(axis=0)


def mce_irl_sample(env, n_traj, *, pi=None, R=None):
    """Sample n_traj trajectories from environment under policy pi.

    pi is assumed to be of shape [t,s,a], like the policy produced by
    `mce_occupancy_measures`."""
    if R is None:
        R = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, R=R)

    demos = {
        "obs": [],
        "states": [],
        "acts": [],
    }

    for _ in range(n_traj):
        obs = env.reset()
        traj_obs = [obs]
        traj_states = [env.cur_state]
        traj_acts = []

        done = False
        t = 0
        while not done:
            pi_now = pi[t, env.cur_state]
            act = np.random.choice(env.n_actions, p=pi_now)
            obs, _, done, _ = env.step(act)

            traj_obs.append(obs)
            traj_states.append(env.cur_state)
            traj_acts.append(act)
            t += 1

        demos["obs"].append(traj_obs)
        demos["states"].append(traj_states)
        demos["acts"].append(traj_acts)

    # stack so that each tensor in the `demos` dict has leading dimension
    # `n_demos` (and same trailing dimensions as before)
    demos = {k: np.stack(v, axis=0) for k, v in demos.items()}

    return demos


def mce_irl(
    env,
    optimiser_tuple,
    rmodel,
    demo_state_om,
    linf_eps=1e-3,
    grad_l2_eps=1e-4,
    max_iter=None,
    print_interval=100,
):
    r"""Discrete MCE IRL.
    Args:
        env (ModelBasedEnv): a tabular MDP.
        optimiser_tuple (tuple): a tuple of `(optim_init_fn, optim_update_fn,
            get_params_fn)` produced by a Jax optimiser.
        rmodel (RewardModel): a reward function to be optimised.
        demo_state_om (np.ndarray): matrix representing state occupancy measure
            for demonstrator.
        linf_eps (float): optimisation terminates if the $l_{\infty}$ distance
            between the demonstrator's state occupancy measure and the state
            occupancy measure for the current reward falls below this value.
        grad_l2_eps (float): optimisation also terminates if the $\ell_2$ norm
            of the MCE IRL gradient falls below this value.
        max_iter (int): absolute max number of iterations to run for.
        print_interval (int or None): how often to log current loss stats
            (using `logging`). None to disable.
    Returns:
        (np.ndarray, np.ndarray): tuple of final parameters found by optimiser
        and state occupancy measure for the final reward function. Note that
        rmodel will also be updated with the latest parameters."""

    obs_mat = env.observation_matrix
    # l_\infty distance between demonstrator occupancy measure (OM) and OM for
    # soft-optimal policy w.r.t current reward (initially set to this value to
    # prevent termination)
    linf_delta = linf_eps + 1
    # norm of the MCE IRL gradient (also set to this value to prevent
    # termination)
    grad_norm = grad_l2_eps + 1
    # number of optimisation steps taken
    t = 0
    assert demo_state_om.shape == (len(obs_mat),)
    opt_init, opt_update, opt_get_params = optimiser_tuple
    rew_params = rmodel.get_params()
    opt_state = opt_init(rew_params)

    while (
        linf_delta > linf_eps
        and grad_norm > grad_l2_eps
        and (max_iter is None or t < max_iter)
    ):
        # get reward predicted for each state by current model, & compute
        # expected # of times each state is visited by soft-optimal policy
        # w.r.t that reward function
        predicted_r, out_grads = rmodel.out_grads(obs_mat)
        _, visitations = mce_occupancy_measures(env, R=predicted_r)
        # gradient of partition function w.r.t parameters; equiv to expectation
        # over states drawn from current imitation distribution of the gradient
        # of the reward function w.r.t its params
        pol_grad = np.sum(visitations[:, None] * out_grads, axis=0)
        # gradient of reward function w.r.t parameters, with expectation taken
        # over states
        expert_grad = np.sum(demo_state_om[:, None] * out_grads, axis=0)
        grad = pol_grad - expert_grad

        # these are just for termination conditions & debug logging
        grad_norm = np.linalg.norm(grad)
        linf_delta = np.max(np.abs(demo_state_om - visitations))
        if print_interval is not None and 0 == (t % print_interval):
            logging.info(
                "Occupancy measure error@iter % 3d: %f (||params||=%f, "
                "||grad||=%f, ||E[dr/dw]||=%f)"
                % (
                    t,
                    linf_delta,
                    np.linalg.norm(rew_params),
                    np.linalg.norm(grad),
                    np.linalg.norm(pol_grad),
                )
            )

        # take a single optimiser step
        opt_state = opt_update(t, grad, opt_state)
        rew_params = opt_get_params(opt_state)
        rmodel.set_params(rew_params)
        t += 1

    return rew_params, visitations
