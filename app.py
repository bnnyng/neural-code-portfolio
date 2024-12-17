import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import helper_functions as F

# =============================
#       SESSION VARIABLES
# =============================

st.session_state.metrics = {}
st.session_state.cell_areas = ['HPC','vmPFC','AMY','dACC','preSMA','VTC']

# =======================
#       IMPORT DATA
# =======================

# ----- Behavioral Data ------
with open("beh.json", "r") as f:
    all_beh_data = json.load(f)["beh"]
st.session_state.beh_data = pd.DataFrame(all_beh_data["data"])
st.session_state.task_data = pd.DataFrame(all_beh_data["task_info"])

# ----- Neural Data ------
st.session_state.neu_data = pd.read_json("neu.json")

# ===================
#       SIDEBAR
# ===================

with st.sidebar:
    with st.container(border=False):
        sidebar_text = """
        #### How to use this notebook

        Words go here.
        """
        st.markdown(sidebar_text)
    with st.container(border=True):
        st.markdown("#### Plot legend")
        st.html(""" 
        <p>
            <span style='color:red;'>●</span> Context <br>
            <span style='color:blue;'>●</span> Outcome <br>
            <span style='color:green;'>●</span> Response <br>
            <span style='color:purple;'>●</span> Stim pair <br>
            <span style='color:orange;'>●</span> Parity 
        </p>
        """)
    with st.expander("Set parameters for analysis"):
        st.write("Not yet implemented.")
    # Changes to sidebar inputs will re-run the entire script
    # TO DO: select brain regions to visualize? Set anaylsis parameters

# TO DO: other tabs with more information


# ==================================
#       RUN GEOMETRIC ANALYSES
# ==================================

@st.cache_data
def run_single_metric(
    metric : str,
    n_resample : int = 5,
    n_perm_inner : int = 1,
    n_samples : list[int] = [15, 15, 15, 15, 15, 10]  ,
):
    return F.run_geometric_analysis(
        metric=metric,
        neu_data=st.session_state.neu_data,
        beh_data=st.session_state.beh_data,
        task_data=st.session_state.task_data,
        n_resample=n_resample,
        n_perm_inner=n_perm_inner,
        n_samples=n_samples
    ) 

# ====================
#       PLOTTING
# ====================

def construct_swarm_for_area(
    title : str,
    metric : str,
    data,
    null_dist,
    y_min : float = 0.4,
    y_max : float = 0.8
):
    """
    Make a swarm plot for inference present vs. absent 
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    F.plot_swarm(
        ax, metric=metric, 
        data=data, 
        null_dist=null_dist,
        xlabels_special=["Absent", "Present"],
        y_min=y_min, y_max=y_max
    )
    if metric == "sd":
        ylabel = "Decoding accuracy (SD)"
    fig.suptitle(title)
    fig.supylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)

# ====================
#       APP BODY
# ====================

st.header("Overall title")

# ----- Explore Data -----
# TO DO:


# ----- Geometric Analyses ------

st.divider()
st.subheader("Geometric analysis of balanced dichotomies")

@st.fragment
def display_single_metric(metric : str):
    if metric not in st.session_state.metrics:
        st.session_state.metrics[f"{metric}"], st.session_state.metrics[f"{metric}_boot"] = run_single_metric(metric)
    data = F.average_sample_data(st.session_state.metrics[f"{metric}"])
    data_boot = F.average_sample_data(st.session_state.metrics[f"{metric}_boot"])

    # Standardize y-scale across all plots
    y_min = min([min(np.array(area_data).flatten()) for area_data in data])
    y_max = min([min(np.array(area_data).flatten()) for area_data in data])

    cols = st.columns(3)
    for i, area_name in enumerate(st.session_state.cell_areas):
        with cols[i % 3]:
            construct_swarm_for_area(
                title=f"{area_name}",
                metric=metric,
                data=[data[0][i], data[1][i]],
                null_dist=[data_boot[0][i], data_boot[1][i]],
                y_min=y_min-0.02,
                y_max=y_max+0.02
            )

# @st.fragment
# def content_sd():
#     # col1, col2 = st.columns(2)
#     # with col1:
#     #     st.write("test")
#     # with col2:
#     #     with st.container(height=300, border=True):
#     #         code = """
#     #             def sd(
#     #                 group_avgs,
#     #                 n_iter : int,
#     #                 n_samples : int,
#     #                 n_folds : int = 5,
#     #                 show_progress : bool = False
#     #             ):
#     #                 _, pos_set, neg_set = define_dichotomies()
#     #                 n_pairs = pos_set.shape[0]
#     #                 n_folds = 5

#     #                 # Matrices for storing performance metrics
#     #                 perf = np.full((n_pairs, n_iter), np.nan)
#     #                 boot = np.full((n_pairs, n_iter), np.nan)

#     #                 # Iterate through all dichotomies
#     #                 for i, (g1, g2) in enumerate(zip(pos_set, neg_set)):
#     #                     # Resample to prevent trial-level bias
#     #                     for j in range(n_iter):
#     #                         training, testing = sample_from_data(
#     #                             group_avgs,
#     #                             n_train=n_samples,
#     #                             n_test=0
#     #                         )
#     #                         train, train_labels, _, _ = prep_regressors(
#     #                             training, testing, g1, g2
#     #                         )
                            
#     #                         # Normalize data and remove empty values
#     #                         scaler = StandardScaler()
#     #                         train_scaled = scaler.fit_transform(train)
#     #                         train_scaled = train_scaled[:, ~np.isnan(train_scaled).any(axis=0)]

#     #                         # Fit linear SVM
#     #                         decoder = LogisticRegression(max_iter=1000, solver="liblinear")
#     #                         y_hat = cross_val_predict(decoder, train_scaled, train_labels, cv=n_folds)
#     #                         perf[i, j] = accuracy_score(train_labels, y_hat)

#     #                         # Compute null distribution
#     #                         shuffled_labels = shuffle(train_labels)
#     #                         y_hat_null = cross_val_predict(decoder, train_scaled, shuffled_labels, cv=n_folds)
#     #                         boot[i, j] = accuracy_score(shuffled_labels, y_hat_null)
#     #                     if show_progress:
#     #                         print(f"Finished dichotomy {i}: {g1}, {g2}")
#     #                 return perf.flatten(), boot.flatten()
#     #         """
#     #         st.code(code, language="python")
#     #     st.caption("This is a test caption.")
#     #     st.write("This is normal text.")
#     display_single_metric("sd")
    
@st.fragment
def content_sd():
    intro = """
    ##### Shattering dimensionality

    **Shattering dimensionality** is defined as the average decoding accuracy across
    all balanced dichotomies. 
    """
    st.markdown(intro)
    display_single_metric("sd")

content_sd()
