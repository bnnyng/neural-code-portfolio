import time
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 42
K_TRIALS = 15

# ===========================
#       DATA PROCESSING     
# ===========================

# ----- Behavioral Data -----

def get_block_num(context : list[int]) -> np.array:
    """
    Define blocks of trials where contexts are alternated by finding where the
    difference between adjacent context values is non-zero.

    Parameters
    ----------
    context : list[int]
        Latent contexts (1 or 2) for each stimulus in the trial.     
    
    Returns
    -------
    np.array
        Array of block numbers for each stimulus.
    """
    transitions = np.where(np.abs(np.diff(context)) > 0)[0] + 1
    block_idx = np.array([[0, *transitions], [*transitions, len(context)]])
    block_nums = np.full(len(context), np.nan)
    for i in range(block_idx.shape[1]):
        block_nums[block_idx[0, i] : block_idx[1, i]] = i 
    return block_nums

def get_instance_num(stim : np.array, block : np.array) -> np.array:
    """
    Get the instance number for the image in each trial block.

    Parameters
    ----------
    stim : np.array
        Stimulus identities for each trial.
    block : np.array
        Array of block numbers for each stimulus.

    Returns
    -------
    np.array
        Array of instance numbers for each stimulus image. 
    """
    instance_nums = np.full(len(stim), np.nan)
    for block_num in np.unique(block):
        instance_dict = {}
        block_idx = np.where(block == block_num)[0]
        for idx in block_idx:
            if stim[idx] not in instance_dict:
                # Reset count for new stimulus
                instance_dict[stim[idx]] = 1
            else:
                instance_dict[stim[idx]] += 1
            instance_nums[idx] = instance_dict[stim[idx]]
    return instance_nums

def get_session_accuracy(
    beh_data : pd.DataFrame,
    task_data : pd.DataFrame
) -> list[np.array]:
    """
    Method for computing session-level accuracy on inference trials (trials in 
    which given stimulus is encountered for the first time after a context 
    switch).

    Parameters
    ----------
    beh_data : pd.DataFrame
        Behavioral data for all sessions.
    task_data : pd.DataFrame
        Task information for each session.

    Returns
    -------
    list[np.array]
        Returns list of arrays of binary values to represent correct vs. 
        incorrect answer. Each array has 5 rows (last trial from previous block
        + 4 stimuli types) and [# blocks - 1] columns (since the first block is
        excluded).
    """
    prop_correct = []
    for i in range(len(task_data)):
        # Get block number
        context = task_data["context"].iloc[i]
        block_nums = get_block_num(context)
        gt = task_data["response_sequence"].iloc[i]
        stim_seq = task_data["stim_sequence"].iloc[i]

        # Find correct answers
        events_code = np.array(beh_data["events"].iloc[i])[:, 1]
        response = np.where(np.isin(events_code, [31, 36]), events_code, np.nan)
        response[response == 36] = 0
        response[response == 31] = 1
        response = response[~np.isnan(response)]
        is_correct = (response == gt)

        # Get instance number for image in each block
        instance_nums = get_instance_num(np.array(stim_seq), block_nums)
        is_correct_first = is_correct[instance_nums == 1]
        first_instance_idx = np.where(instance_nums == 1)[0]
        block_first_instance = block_nums[first_instance_idx]

        # Get accuracy of all first instances; pad with NaN if lengths mismatched
        idx = [
            np.where(block_first_instance == block)[0]
            for block in np.unique(block_first_instance)
        ]
        temp = [is_correct_first[i] for i in idx]
        max_len = max(len(instances) for instances in temp)
        temp = [
            np.pad(instances, (0, max_len - len(instances)), constant_values=np.nan)
            for instances in temp
        ]
        is_correct_first = np.concatenate(temp)

        # Reshape to stack accuracy across blocks
        is_correct_first = is_correct_first.reshape(
            len(np.unique(stim_seq)), int(np.max(block_nums).T) + 1
        )
        
        # Append last trial from previous block for each stimulus
        last_trials_idx = [
            np.where(block_nums == block)[0][-1]
            for block in np.unique(block_nums)
        ]
        is_correct_last = is_correct[last_trials_idx]

        # Exclude first block
        prop_correct.append(np.vstack([is_correct_last, is_correct_first])[:, 1:])
    return prop_correct

def test_inference_trials(
    beh_data : pd.DataFrame,
    task_data : pd.DataFrame
) -> dict:
    """
    Parameters
    ----------
    beh_data : pd.DataFrame
    task_data : pd.DataFrame

    Returns
    -------
    dict
        Dictionary with statistical significance for baseline/inference trials
        and proportion of correct responses on inference trials.
    """
    # Get session-level accuracy for each trial
    prop_correct = get_session_accuracy(beh_data, task_data)

    baseline = np.full(len(prop_correct), np.nan)
    inference = np.full(len(prop_correct), np.nan)
    inf_perf = np.full(len(prop_correct), np.nan)

    for i, X in enumerate(prop_correct):
        # First inference trial in third column
        inf_trials = X[:, 2]
        valid_trials = np.sum(~np.isnan(inf_trials))

        # Check significance of behavior against chance (0.5)
        baseline[i] = 1 - binom.cdf(np.sum(X[:, 0]), len(X[:, 0]), 0.5)
        inference[i] = 1 - binom.cdf(np.nansum(inf_trials), valid_trials, 0.5)
        inf_perf[i] = np.nansum(inf_trials) / valid_trials

    # TO DO: why are indices selected this way?
    return {"baseline" : baseline, "inference": inference, "inf_perf" : inf_perf}

def split_sessions(
    neu_data : pd.DataFrame,
    beh_data : pd.DataFrame,
    task_data : pd.DataFrame,
    p : float = 0.05
):
    """
    Wrapper for computing significance of session-level inference behavior.

    Parameters
    ----------
    neu_data : pd.DataFrame
        DataFrame with each row corresponding to all trial data for a single 
        neuron.
    beh_data : pd.DataFrame
        Behavioral data for all sessions.
    task_data : pd.DataFrame
        Task information for each session.
    p : float
        p-value for statistical significance.
    
    Returns
    -------
    dict
        Dictionary of sessions grouped by inference presence/absence.
    """
    session_names = beh_data["sessionID"].to_list()
    perf_dict = test_inference_trials(beh_data, task_data)

    # Determine inference present/absent groups using significance
    inf_absent = [
        name for i, name in enumerate(session_names)
        if perf_dict["inference"][i] > p and perf_dict["baseline"][i] < p
    ]
    inf_present = [
        name for i, name in enumerate(session_names)
        if perf_dict["inference"][i] < p and perf_dict["baseline"][i] > p
    ]
    
    all_sessions = neu_data["sessionID"]
    return {
        "absent" : inf_absent,
        "present" : inf_present,
        "absent_idx" : [i for i, name in enumerate(all_sessions) if name in inf_absent],
        "present_idx" : [i for i, name in enumerate(all_sessions) if name in inf_present],
        "sessions" : session_names
    }

# ----- Neural Data -----

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

def define_cell_area_groups(neu_data : pd.DataFrame) -> dict:
    area_order = ['HPC','vmPFC','AMY','dACC','preSMA','VTC']
    idx_order = [3, 1, 4, 2, 9, 12]
    cellinfo = neu_data.cellinfo.to_numpy()
    cell_area_groups = {}
    for i, area in enumerate(area_order):
        idx = np.where(cellinfo == idx_order[i])[0]
        cell_area_groups[area] = idx
    return cell_area_groups

# ================================
#       BALANCED DICHOTOMIES
# ================================

def define_dichotomies() -> tuple[dict, np.array, np.array]:
    """
    Define the set of all possible balanced dichotomies by all possible ways
    that eight unique task conditions can be split into two groups of four 
    conditons each. There are (8 choose 4)/2 = 35 possible balanced dichotomies.

    Within each context (1 or 2), each stimulus category (A, B, C, D) is uniquely
    specified by its response-outcome pair (left or right, high or low reward, 
    respectively).

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
    # TO DO: where is parity?
    interpretations = {
        0 : "context",
        9 : "outcome",
        11 : "AB vs CD",
        20 : "response",
        23 : "AC vs BD", 
        31 : "AD vs BC"
    }
    pos_set = np.array([
        [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 3, 7], [1, 2, 3, 8],
        [1, 2, 4, 5], [1, 2, 4, 6], [1, 2, 4, 7], [1, 2, 4, 8], [1, 2, 5, 6],
        [1, 2, 5, 7], [1, 2, 5, 8], [1, 2, 6, 7], [1, 2, 6, 8], [1, 2, 7, 8],
        [1, 3, 4, 5], [1, 3, 4, 6], [1, 3, 4, 7], [1, 3, 4, 8], [1, 3, 5, 6],
        [1, 3, 5, 7], [1, 3, 5, 8], [1, 3, 6, 7], [1, 3, 6, 8], [1, 3, 7, 8],
        [1, 4, 5, 6], [1, 4, 5, 7], [1, 4, 5, 8], [1, 4, 6, 7], [1, 4, 6, 8],
        [1, 4, 7, 8], [1, 5, 6, 7], [1, 5, 6, 8], [1, 5, 7, 8], [1, 6, 7, 8]
    ])
    neg_set = np.array([
        [5, 6, 7, 8], [4, 6, 7, 8], [4, 5, 7, 8], [4, 5, 6, 8], [4, 5, 6, 7],
        [3, 6, 7, 8], [3, 5, 7, 8], [3, 5, 6, 8], [3, 5, 6, 7], [3, 4, 7, 8],
        [3, 4, 6, 8], [3, 4, 6, 7], [3, 4, 5, 8], [3, 4, 5, 7], [3, 4, 5, 6],
        [2, 6, 7, 8], [2, 5, 7, 8], [2, 5, 6, 8], [2, 5, 6, 7], [2, 4, 7, 8],
        [2, 4, 6, 8], [2, 4, 6, 7], [2, 4, 5, 8], [2, 4, 5, 7], [2, 4, 5, 6],
        [2, 3, 7, 8], [2, 3, 6, 8], [2, 3, 6, 7], [2, 3, 5, 8], [2, 3, 5, 7],
        [2, 3, 5, 6], [2, 3, 4, 8], [2, 3, 4, 7], [2, 3, 4, 6], [2, 3, 4, 5]
    ]) 
    return interpretations, pos_set - np.ones_like(pos_set), neg_set - np.ones_like(neg_set)

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
    unique_values = [sorted(cell_data[var].unique()) for var in var_names]
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
    group_avgs = pd.DataFrame()
    for i in select:
        # Select only correct trials for given neuron
        cell_data = get_cell_array(neu_data=neu_data, cell_idx=i)
        valid_data = cell_data[cell_data["iscorrect"] == True]
        
        # Make trial-level labels according to binary task variables
        groups = make_variable_groups(
            cell_data=valid_data,
            var_names=["context", "reward", "response"]
        )     

        firing_rate = valid_data["fr_stim"].values
        group_avg = []
        for combo in groups.columns:
            # Get boolean mask for inclusion in current group
            group_firing_rate = firing_rate[groups[combo]]
            # Check that neuron has enough samples of the current type
            if len(group_firing_rate) > sample_thr:
                group_avg.append(group_firing_rate)
            if combo not in group_avgs.columns:
                group_avgs[combo] = None
        # Only append data for neurons with enough trials in all types
        if len(group_avg) == len(groups.columns):
            group_avgs.loc[len(group_avgs)] = group_avg
    return group_avgs

# =====================================
#       SHATTERING DIMENSIONALITY
# =====================================

def sample_from_data(
    group_avgs : pd.DataFrame,
    n_train : int,
    n_test : int,
    replace : bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame()
    testing = pd.DataFrame()

    for combo in group_avgs.columns:
        cell_train = []
        cell_test = []
        for cell_data in group_avgs[combo].values:
            train_idx = np.random.choice(len(cell_data), n_train, replace=replace)
            test_idx = np.random.choice(
                len(np.delete(cell_data, train_idx)), n_test, replace=replace
            )
            cell_train.append(cell_data[train_idx])
            cell_test.append(cell_data[test_idx])
        training[combo] = cell_train
        testing[combo] = cell_test
    return training, testing

def prep_regressors(
    training : pd.DataFrame,
    testing : pd.DataFrame,
    g1 : list[int],
    g2 : list[int]
):
    """
    """
    train1 = np.vstack([
        np.vstack(training.iloc[i, g1])
        for i in range(len(training))
    ])
    train2 = np.vstack([
        np.vstack(training.iloc[i, g2])
        for i in range(len(training))
    ])
    train_labels = np.concatenate([
        np.ones(len(train1)), -1 * np.ones(len(train2))
    ])

    test1 = np.vstack([
        np.vstack(testing.iloc[i, g1])
        for i in range(len(testing))
    ])
    test2 = np.vstack([
        np.vstack(testing.iloc[i, g2])
        for i in range(len(testing))
    ])
    test_labels = np.concatenate([
        np.ones(len(test1)), -1 * np.ones(len(test2))
    ])

    train, test = np.vstack([train1, train2]), np.vstack([test1, test2])
    return train, train_labels, test, test_labels

def sd(
    group_avgs,
    n_iter : int,
    n_samples : int,
    n_folds : int = 5,
    show_progress : bool = False
):
    """
    Method for performing shattering dimensionality analysis for a group of 
    cells. Shattering dimensionality is defined as the average decoding accuracy
    across all balanced dichotomies.

    Parameters
    ----------
    group_avgs : pd.DataFrame
        Data for averaged neuron firing rates across all combinations of task
        variables. Each row represents a single neuron, and columns correspond 
        to different groupings of task variables. The DataFrame is generated by
        the `construct_regressors` method.
    n_iter : int
        Number of iterations of bootstrap re-sampling to perform.
    n_samples : int
        Number of trials of each condition to sample.
    n_folds : int
        Number of folds to use in cross-validation.
    show_progress : bool
        Whether to print the training progress.

    Returns
    -------
    np.array
        TO DO
    np.array

    """
    _, pos_set, neg_set = define_dichotomies()
    n_pairs = pos_set.shape[0]
    n_folds = 5

    # Matrices for storing performance metrics
    perf = np.full((n_pairs, n_iter), np.nan)
    boot = np.full((n_pairs, n_iter), np.nan)

    # Iterate through all dichotomies
    for i, (g1, g2) in enumerate(zip(pos_set, neg_set)):
        # Resample to prevent trial-level bias
        for j in range(n_iter):
            training, testing = sample_from_data(
                group_avgs,
                n_train=n_samples,
                n_test=0
            )
            train, train_labels, _, _ = prep_regressors(
                training, testing, g1, g2
            )
            
            # Normalize data and remove empty values
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            train_scaled = train_scaled[:, ~np.isnan(train_scaled).any(axis=0)]

            # Fit linear SVM
            decoder = LogisticRegression(max_iter=1000, solver="liblinear")
            y_hat = cross_val_predict(decoder, train_scaled, train_labels, cv=n_folds)
            perf[i, j] = accuracy_score(train_labels, y_hat)

            # Compute null distribution
            shuffled_labels = shuffle(train_labels)
            y_hat_null = cross_val_predict(decoder, train_scaled, shuffled_labels, cv=n_folds)
            boot[i, j] = accuracy_score(shuffled_labels, y_hat_null)
        if show_progress:
            print(f"Finished dichotomy {i}: {g1}, {g2}")
    return perf.flatten(), boot.flatten()

# ====================
#       PLOTTING      
# ====================

def average_sample_data(sample_data : list):
    avg_data = []
    for session_type in sample_data:
        type_data = []
        for area_data in session_type:
            type_data.append(np.mean(np.array(area_data), axis=0))
        avg_data.append(type_data)
    return avg_data

def plot_swarm(
    ax, 
    metric : str,
    data : list, 
    null_dist : list = None,
    idx_special : list = [0, 9, 20, 23, 28],
    idx_labels : list = ["Context", "Outcome", "Response", "Stim pair", "Parity"],
    xlabels_special : list = [],
    connection_map : list = None,
    clrs_list : list = None,
    y_min : float = 0.4,
    y_max : float = 0.9
):
    """
    Swarm plot method for results of geometric analysis.
    """
    if connection_map is None:
        connection_map = [0, 1] * (len(data) // 2)
    if clrs_list is None:
        clrs_list = sns.color_palette("Set1", len(idx_special))

    # Plot null distribution as filled rectangles
    if null_dist is not None:
        offset = 0.35
        for i, null_data in enumerate(null_dist):
            lims = np.percentile(null_data, [5, 95])
            ax.fill(
                 [i - offset, i + offset, i + offset, i - offset],
                 [lims[0], lims[0], lims[1], lims[1]], color="gray", alpha=0.2
            )
    
    # Set plot attributes based on metric
    facecolors = "white" if metric == "sd" else "gray"
    marker = "^" if metric == "ps" else "o"
    linewidth = 2 if metric =="ps" else 4

    # Plot non-special points
    to_plot = []
    marker_size = 80
    for i in range(len(data)):
        if idx_special:
            to_plot.append(data[i][idx_special])
            data[i] = np.delete(data[i], idx_special)      
        # TO DO: jitter?
        ax.scatter(
            np.full(len(data[i]), i), data[i],
            s=marker_size, marker=marker,
            edgecolors="gray", facecolors=facecolors,
            linewidth=linewidth, alpha=0.7
        )
    
    # Plot special points and connections
    for i in range(len(data)):
        for j in range(len(idx_special)):
            if i == 1:
                ax.plot(
                    [i-1, i], (to_plot[i-1][j], to_plot[i][j]),
                    linewidth=2.5, color=clrs_list[j]
                )
            ax.scatter(
                i, to_plot[i][j],
                s=marker_size+40, marker=marker,
                edgecolors=clrs_list[j], facecolors=facecolors,
                linewidth=linewidth, 
                label=(idx_labels[j] if i == 1 else None)
            )
    # ax.legend(loc="center right", bbox_to_anchor=(1, 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xlabels_special)

# =============================
#      GEOMETRIC ANALYSES      
# =============================

def run_geometric_analysis(
    metric : str,
    neu_data : pd.DataFrame,
    beh_data : pd.DataFrame,
    task_data : pd.DataFrame,
    n_resample : int = 5,
    n_perm_inner : int = 1,
    n_samples : list[int] = [15, 15, 15, 15, 15, 10]  ,
    show_progress : bool = False
):
    # Step 1: Compute session-level inference performance
    sess = split_sessions(neu_data, beh_data, task_data)
    inf_absent, inf_present = sess["absent"], sess["present"]
    absent_idx, present_idx = sess["absent_idx"], sess["present_idx"]
    idx_sets = [absent_idx, present_idx]
    session_names = sess["sessions"]
    all_sessions = neu_data["sessionID"].to_list()

    # Step 2: Aggregate neurons by area
    cell_area_groups = define_cell_area_groups(neu_data)

    # Step 3: Run geometric analyses
    data_ = [
        [[] for _ in range(len(cell_area_groups))],
        [[] for _ in range(len(cell_area_groups))]
    ]
    data_boot = [
        [[] for _ in range(len(cell_area_groups))],
        [[] for _ in range(len(cell_area_groups))]
    ]

    for i, (area_name, area_idx) in enumerate(cell_area_groups.items()):
        start_time = time.time()
        for j, idx_set in enumerate(idx_sets):
            if show_progress:
                curr_set = "inference absent" if j == 0 else "inference present"
                print(f"Running analyses for {curr_set} trials over {n_samples[i]} samples...")
            curr_idx = np.intersect1d(area_idx, idx_set)
            for _ in range(n_resample):
                group_avgs = construct_regressors(neu_data, n_samples[i], curr_idx)
                if metric == "sd":
                    t_1, t_2 = sd(group_avgs, n_perm_inner, n_samples[i])
                    data_[j][i].append(t_1)
                    data_boot[j][i].append(t_2)
                    if show_progress:
                        ("SD complete.")
        end_time = time.time()
        if show_progress:
            print(f"Analyses complete for area {area_name}. Time: {end_time - start_time:.6f} seconds.")
    return data_, data_boot