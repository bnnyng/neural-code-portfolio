import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

import helper_functions as F

# =============================
#       SESSION VARIABLES
# =============================

# TO DO: incorporate async

st.session_state.metrics = {}
st.session_state.cell_areas = ['HPC','vmPFC','AMY','dACC','preSMA','VTC']
st.session_state.analysis_params = {
    "n_resample" : 5,
    "n_perm_inner" : 1,
    "n_samples" : [15, 15, 15, 15, 15, 10],
    "n_iter_boot" : 10  
}
st.session_state.analysis_params_update = False

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

def change_params():
    st.session_state.analysis_params_update = True
    for key, value in st.session_state.analysis_params.items():
        st.write(f"{key}: {value}")

with st.sidebar:
    with st.container(border=False):
        # sidebar_text = """
        # ### How to use this app
        # """
        # st.markdown(sidebar_text)
        caption = """
            Source code for the Python implementation used in this project is available 
            at the following repository: [bnnyng/neural-code-portfolio](https://github.com/bnnyng/neural-code-portfolio). 
            
            Original MatLab code from the paper is available at: [osf.io/qpt8f/](https://osf.io/qpt8f/)
        """
        st.caption(caption)
    with st.expander("Legend for geometric analysis plots"):
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
        st.caption("""
        Warning: Changes to parameter values will re-run the entire script.
                   
        Previously used values are cached, but new values require analyses to be performed from scratch.
        """)
        st.session_state.analysis_params["n_resample"] = st.number_input(
            label="Number of times to resample analysis in each brain area:",
            min_value=1, max_value=10000, value=5, on_change=change_params
        )
        st.session_state.analysis_params["n_perm_inner"] = st.number_input(
            label="Number of iterations for geometric analyses:",
            min_value=1, max_value=10000, value=1, on_change=change_params
        )
        st.session_state.analysis_params["n_iter_boot"] = st.number_input(
            label="Number of iterations for computing null distributions:",
            min_value=5, max_value=10000, value=10, on_change=change_params
        )

# ==================================
#       RUN GEOMETRIC ANALYSES
# ==================================

@st.cache_data
def run_single_metric(
    metric : str,
    n_resample : int = 5,
    n_perm_inner : int = 1,
    n_samples : list[int] = [15, 15, 15, 15, 15, 10],
    n_iter_boot : int = 1000
):
    return F.run_geometric_analysis(
        metric=metric,
        neu_data=st.session_state.neu_data,
        beh_data=st.session_state.beh_data,
        task_data=st.session_state.task_data,
        n_resample=n_resample,
        n_perm_inner=n_perm_inner,
        n_iter_boot=n_iter_boot,
        n_samples=n_samples,
        show_progress = True
    ) 

# ====================
#       PLOTTING
# ====================


# ----- Neural State Space -----



# ----- Plot Geometric Analyses -----

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
    if metric == "ccgp":
        ylabel = "CCGP"
    fig.suptitle(title)
    fig.supylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)

# ====================
#       APP BODY
# ====================

st.header("Neural Code Final Portfolio: Component 2")

def content_introduction():
    intro = """
    This project replicates the key methods and analyses of representational 
    geometries for neuron populations conducted in [1]. In particular, the project
    source code is a custom Python implementation based on descriptions from the 
    original paper and example MatLab scripts included in the open-source dataset.
    
    The app consists of the following sections:
    * **Explore data:** a walkthrough of orignal data and processing methods used
    for geometric analyses. DataFrames are embedded for 
    * **Visualize neuron state space:** a lower-dimensional visualization of neural
    responses in the hippocampus.
    * Geometric analysis of balanced dichotomies: 

    The purpose of this app is to unpack the computational analysis being
    performed in a cutting-edge neuroscience paper to be accessible to undergraduates
    and other non-experts. Thus, the focus of this project is on **exposition
    of methods**, rather than the underlying neuroscience theory that is covered
    by the original paper. This app also should not be viewed as a full attempt
    to replicate the results of [1] since default parameter values are low in order to keep
    computations feasible (see left sidebar), although the underlying source code
    is available for that purpose.
    """
    st.markdown(intro)
    citation = """
    [1] Courellis, H.S., Minxha, J., Cardenas, A.R. et al. Abstract representations 
    emerge in human hippocampal neurons during inference. Nature 632, 841–849 (2024). 
    [doi.org/10.1038/s41586-024-07799-x](https://doi.org/10.1038/s41586-024-07799-x)
    """
    st.caption(citation)

content_introduction()

# TO DO: Concurrent.futures to thread differently

# ----- Explore Data -----
st.divider()
st.subheader("Explore data")

@st.fragment
def content_data():
    cell_level = """
    ##### Trial-level firing rates for each neuron

    The raw data used for geometric analyses consists of binned spike counts for 
    all neurons across the six brain regions studied in the paper. The following trial-level info
    is available for each neuron. Each set of trials belongs to a single experimental
    session, and each cell has between 180 and 320 trials.
    """
    st.markdown(cell_level)
    cell_idx = st.number_input(
        label="Enter an a cell index to view trial data (minimum 0, maximum 2963):",
        min_value=0, max_value=2963, value=0
    )
    sample_cell_data = F.get_cell_array(
        st.session_state.neu_data,
        cell_idx
    )
    st.dataframe(sample_cell_data)
    

content_data()

# ----- Visualize Neuron State Space -----

st.divider()
st.subheader("Visualize neuron state space")

@st.cache_data
def content_state_space():
    intro = """
    
    """
    st.markdown(intro)

    fig_pres = F.plot_neu_state_space(
        st.session_state.neu_data,
        st.session_state.beh_data,
        st.session_state.task_data,
        inf="present"
    )
    fig_abs = F.plot_neu_state_space(
        st.session_state.neu_data,
        st.session_state.beh_data,
        st.session_state.task_data,
        inf="absent"
    )

    col1, col2 = st.columns(2)
    with col1: 
        st.pyplot(fig_abs)
    with col2:
        st.pyplot(fig_pres)
    caption = """
    Hippocampal population response during stimulus period in inference present 
    and absent sessions. Pairwise disimilarities between different task conditions
    are visualized in a lower-dimensional space using multidimensional scaling (MDS). Points correspond to stimulus-context combinations (i.e., the 8 task conditions).
    Lines connect the same stimuli across contexts. These figures replicate 2(j) and 3(i) from the original paper.
    """
    st.caption(caption)

content_state_space()


# ----- Geometric Analyses ------

st.divider()
st.subheader("Geometric analysis of balanced dichotomies")

@st.fragment
def display_single_metric(metric : str):
    if metric not in st.session_state.metrics or st.session_state.analysis_params_update:
        st.session_state.metrics[f"{metric}"], st.session_state.metrics[f"{metric}_boot"] = run_single_metric(
            metric=metric,
            n_resample=st.session_state.analysis_params["n_resample"],
            n_perm_inner=st.session_state.analysis_params["n_perm_inner"],
            n_iter_boot=st.session_state.analysis_params["n_iter_boot"],
            n_samples=st.session_state.analysis_params["n_samples"]
        )
        st.session_state.analysis_params_update = False
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
    
@st.fragment
def content_sd(update_params : bool = False):
    intro = """
    ##### Shattering dimensionality

    **Shattering dimensionality** is defined as the average decoding accuracy across
    all balanced dichotomies. 
    """
    st.markdown(intro)
    display_single_metric("sd")

@st.fragment
def content_ccgp(update_params : bool = False):
    intro = """ 
    ##### Cross-condition generalization performance

    **Cross-condition generalization performance (CCGP)** is a measurement of the 
    extent to which a decoder trained on one set of task conditions can generalize
    to another set. 
    """
    st.markdown(intro)
    display_single_metric("ccgp")

@st.fragment
def content_ps(update_params : bool = False):
    intro = """ 
    ##### Parallelism score 

    **Parallelism score** is...
    """
    st.markdown(intro)
    display_single_metric("ps")

content_sd(st.session_state.analysis_params_update)
# content_ps(st.session_state.analysis_params_update)
# content_ccgp()


