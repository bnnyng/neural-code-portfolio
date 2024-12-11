import numpy as np
import pandas as pd
import itertools

def get_cell_array(
    neu_data : pd.DataFrame,
    cell_idx : int
 ) -> pd.DataFrame:
    cell_array_dict = {}
    for row_dict in neu_data["array"][cell_idx]:
        for key, value in row_dict.items():
            if key not in cell_array_dict:
                cell_array_dict[key] = [value]
            else:
                cell_array_dict[key].append(value)
    cell_array_data = pd.DataFrame(cell_array_dict)
    return cell_array_data


def define_dichotomies() -> tuple[dict, np.array, np.array]:
    """
    Define the set of all possible balanced dichotomies by all possible ways
    that eight unique task conditions can be split into two groups of four 
    conditons each. There are (8 choose 4)/2 = 35 possible balanced dichotomies.
    
    Conditions are defined as follows [#: stimulus, response, outcome, context]:
    - 1: C, left, 5, 1
    - 2: D, right, 5, 1
    - 3: A, left, 25, 1
    - 4: B, right, 25, 1
    - 5: D, left 5, 2
    - 6: A, right, 5, 2
    - 7: B, left, 25, 2
    - 8: C, right, 25, 2

    Returns
    -------
    dict
        Dictionary of (index, name) pair for dichotomies with clear 
        interpretations with respect to task condition.
    np.array
        First class of dichotomies.
    np.array
        Second class of dichotomies.
    """
    interpretations = {
        0 : "context",
        9 : "outcome",
        11 : "AB vs CD",
        20 : "response",
        23 : "AC vs BD", 
        31 : "AD vs BC"
    }
    class1 = np.array([
        [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 3, 7], [1, 2, 3, 8],
        [1, 2, 4, 5], [1, 2, 4, 6], [1, 2, 4, 7], [1, 2, 4, 8], [1, 2, 5, 6],
        [1, 2, 5, 7], [1, 2, 5, 8], [1, 2, 6, 7], [1, 2, 6, 8], [1, 2, 7, 8],
        [1, 3, 4, 5], [1, 3, 4, 6], [1, 3, 4, 7], [1, 3, 4, 8], [1, 3, 5, 6],
        [1, 3, 5, 7], [1, 3, 5, 8], [1, 3, 6, 7], [1, 3, 6, 8], [1, 3, 7, 8],
        [1, 4, 5, 6], [1, 4, 5, 7], [1, 4, 5, 8], [1, 4, 6, 7], [1, 4, 6, 8],
        [1, 4, 7, 8], [1, 5, 6, 7], [1, 5, 6, 8], [1, 5, 7, 8], [1, 6, 7, 8]
    ])
    class2 = np.array([
        [5, 6, 7, 8], [4, 6, 7, 8], [4, 5, 7, 8], [4, 5, 6, 8], [4, 5, 6, 7],
        [3, 6, 7, 8], [3, 5, 7, 8], [3, 5, 6, 8], [3, 5, 6, 7], [3, 4, 7, 8],
        [3, 4, 6, 8], [3, 4, 6, 7], [3, 4, 5, 8], [3, 4, 5, 7], [3, 4, 5, 6],
        [2, 6, 7, 8], [2, 5, 7, 8], [2, 5, 6, 8], [2, 5, 6, 7], [2, 4, 7, 8],
        [2, 4, 6, 8], [2, 4, 6, 7], [2, 4, 5, 8], [2, 4, 5, 7], [2, 4, 5, 6],
        [2, 3, 7, 8], [2, 3, 6, 8], [2, 3, 6, 7], [2, 3, 5, 8], [2, 3, 5, 7],
        [2, 3, 5, 6], [2, 3, 4, 8], [2, 3, 4, 7], [2, 3, 4, 6], [2, 3, 4, 5]
    ]) 
    return interpretations, class1, class2

def make_variable_groups(
    cell_data : pd.DataFrame,
    var_names : list[str]
) -> pd.DataFrame:
    """
    Create groups of all possible combinations of values for the given 
    variables.

    Parameters
    ----------
    cell_data : pd.DataFrame
        Data for a single neuron, where each row represents one trial.
    var_names : list[str]
        List of DataFrame columns to consider.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row represents a trial and each column represents 
        membership in a group.
    """
    unique_values = [cell_data[var].unique() for var in var_names]
    combinations = list(itertools.product(*unique_values))
    group_names = [
        "_".join([f"{var}_{value}" for var, value in zip(var_names, combo)])
        for combo in combinations
    ]
    groups = pd.DataFrame(np.zeros(
        (len(cell_data), len(combinations)), dtype=bool
    ), columns=group_names)
    for i, combo in enumerate(combinations):
        # Set combination membership for all trials in a single column
        condition = np.all([
            cell_data[var] == value for var, value in zip(var_names, combo)
        ], axis=0)
        groups[group_names[i]] = condition
    return groups

def construct_regressors(
    neu_data : pd.DataFrame,
    sample_thr : int,
    select : list
) -> pd.DataFrame:
    """
    Method for balancing neuron counts between inference absent (ia) and 
    inference present (ip) groups.
    
    Parameters
    ----------
    neu_data : pd.DataFrame
        DataFrame with each row correpsonding to all trial data for a single 
        neuron.
    sample_thr : int
        Minimum number of correct trials of each type to retain neurons.
    select : list
        Neuron indices to include.

    Returns
    -------
    pd.DataFrame
        DataFrame of neurons that clear the trial count threshold. Each row
        corresponds to a neuron, and each column corresponds to the average 
        firing rate for a combination of task variables.
    """
    group_avgs = []
    for i in select:
        # Select only correct trials for given neuron
        cell_data = get_cell_array(neu_data=neu_data, cell_idx=i)
        valid_data = cell_data[cell_data["iscorrect"] == True]
        
        # Make trial-level labels according to binary task variables
        groups = make_variable_groups(
            cell_data=valid_data,
            var_names=["reward", "response", "context"]
        )     

        firing_rate = valid_data["fr_stim"].values
        group_avg = []
        for combo in groups.columns:
            # Get boolean mask for inclusion in current group
            group_firing_rate = firing_rate[groups[combo]]
            # Check that neuron has enough samples of the current type
            if len(group_firing_rate) > sample_thr:
                group_avg.append(group_firing_rate.mean())
        # Only append data for neurons with enough trials in all types
        if len(group_avg) == len(groups.columns):
            group_avgs.append(group_avg)
    return pd.DataFrame(np.array(group_avgs), columns=groups.columns)