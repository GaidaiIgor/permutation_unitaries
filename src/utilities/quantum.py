""" Quantum-related utilities """
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from numpy import ndarray
from pysat.examples.hitman import Hitman


def get_average_neighbors(bases: ndarray) -> float:
    """ Returns average number of neighbor bases in the given array of bases. """
    if bases.shape[0] == 0:
        return 0

    neighbors = 0
    for basis1, basis2 in itertools.combinations(bases, 2):
        if np.sum(basis1 != basis2) == 1:
            neighbors += 2
    return neighbors / bases.shape[0]


def get_different_inds(basis_1: Sequence, basis_2: Sequence, ignore_ind: int) -> list[int]:
    """ Returns different indices between two bases, ignoring ignore_ind. """
    return [ind for ind in range(len(basis_1)) if ind != ignore_ind and basis_1[ind] != basis_2[ind]]


def solve_minimum_hitting_set(sets: list[list[int]]) -> list[int]:
    """ Finds the smallest set of integers that overlaps with all given sets. """
    hitman = Hitman()
    for set in sets:
        hitman.hit(set)
    solution = hitman.get()
    return solution


def find_min_control_set(existing_states: ndarray, target_state_ind: int, interaction_ind: int) -> list[int]:
    """
    Finds minimum set of control necessary to select the target state and its destination interacting via interaction_ind.
    :param existing_states: List of states with non-zero amplitudes.
    :param target_state_ind: Index of the target state in the existing_states.
    :param interaction_ind: Index of target qubit for the controlled operation (to exclude from consideration for the control set).
    :return: Minimum set of control indices necessary to select the target state.
    """
    origin = existing_states[target_state_ind, :]
    destination = origin.copy()
    destination[interaction_ind] ^= 1
    different_inds = [get_different_inds(state, origin, interaction_ind) for state in existing_states if not (np.all(state == origin) or np.all(state == destination))]
    return solve_minimum_hitting_set(different_inds)


def find_min_control_set_2(existing_states: list[list[int]] | ndarray, target_state_ind: int, candidate_interaction_inds: Sequence) -> (ndarray, ndarray, int):
    """ Higher level wrapper around find_min_control_set. Handles state transformation under CX conjugation and iterates over candidate interaction indices to find the best.
    Returns control indices, values and selected interaction index. """
    result = None
    for interaction_ind in candidate_interaction_inds:
        transformed_states = np.array(existing_states)
        cx_target_inds = [ind for ind in candidate_interaction_inds if ind != interaction_ind]
        transformed_states[np.ix_(transformed_states[:, interaction_ind] == 1, cx_target_inds)] ^= 1
        control_inds = find_min_control_set(transformed_states, target_state_ind, interaction_ind)
        if result is None or len(result[0]) > len(control_inds):
            control_vals = transformed_states[target_state_ind, control_inds]
            result = (np.array(control_inds), control_vals, interaction_ind)
    return result


def get_cx_cost_cx(num_controls: int) -> int:
    """ Returns the number of CX gates in the decomposition of multi-controlled Rx gate with the specified number of controls. """
    cx_by_num_controls = [0, 1, 6, 14, 36, 56, 80, 104]
    if num_controls < len(cx_by_num_controls):
        return cx_by_num_controls[num_controls]
    else:
        return cx_by_num_controls[-1] + (num_controls - len(cx_by_num_controls) - 1) * 16
