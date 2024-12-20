import streamlit as st
import numpy as np
import pandas as pd
import json
import random
import inspect
import matplotlib.pyplot as plt

import helper_functions as F

RANDOM_STATE = 42
MAX_IDX = 2693

# =============================
#       SESSION VARIABLES
# =============================

st.session_state.metrics = {
    "sd" : np.load("sample-results/sd_allAreas.npy", allow_pickle=True),
    "sd_boot" :  np.load("sample-results/sd_allAreas_boot.npy", allow_pickle=True),
    "ccgp" : np.load("sample-results/ccgp_allAreas.npy", allow_pickle=True),
    "ccgp_boot" :  np.load("sample-results/ccgp_allAreas_boot.npy", allow_pickle=True),
    "ps" : np.load("sample-results/ps_allAreas.npy", allow_pickle=True),
    "ps_boot" :  np.load("sample-results/ps_allAreas_boot.npy", allow_pickle=True),
}
st.session_state.cell_areas = ['HPC','vmPFC','AMY','dACC','preSMA','VTC']
st.session_state.params = {
    "n_resample" : 5,
    "n_perm_inner" : 1,
    "n_samples_list" : [15, 15, 15, 15, 15, 10],
    "n_iter_boot" : 10,
    "n_samples" : 15  
}
st.session_state.params_update = False

# =======================
#       IMPORT DATA
# =======================

# ----- Behavioral Data ------
with open("beh.json", "r") as f:
    all_beh_data = json.load(f)["beh"]
st.session_state.params["beh_data"] = pd.DataFrame(all_beh_data["data"])
st.session_state.params["task_data"] = pd.DataFrame(all_beh_data["task_info"])

# ----- Neural Data ------
st.session_state.params["neu_data"] = pd.read_json("neu.json")

# ===================
#       SIDEBAR
# ===================

def change_params():
    # Currently deactivated
    st.session_state.params_update = False # True

with st.sidebar:
    with st.container(border=False):
        # sidebar_text = """
        # ### How to use this app
        # """
        # st.markdown(sidebar_text)
        caption = """
            Source code for the Python implementation used in this project is available 
            at the following repository, as well as a notebook explaining
            data pre-processing methods: [bnnyng/neural-code-portfolio](https://github.com/bnnyng/neural-code-portfolio). 
            
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
    with st.expander("Set parameters for analysis (inactive)"):
        st.write("This feature is currently inactive.")
        st.caption("""
        Warning: Changes to parameter values will re-run the entire script.
                   
        Previously used values are cached, but new values require analyses to be performed from scratch.
        """)
        st.session_state.params["n_resample"] = st.number_input(
            label="Number of times to resample analysis in each brain area:",
            min_value=1, max_value=10000, value=5, on_change=change_params
        )
        st.session_state.params["n_perm_inner"] = st.number_input(
            label="Number of iterations for geometric analyses:",
            min_value=1, max_value=10000, value=1, on_change=change_params
        )
        st.session_state.params["n_iter_boot"] = st.number_input(
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
        neu_data=st.session_state.params["neu_data"],
        beh_data=st.session_state.params["beh_data"],
        task_data=st.session_state.params["task_data"],
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
        ylabel = "Decoding Accuracy (SD)"
    if metric == "ccgp":
        ylabel = "CCGP"
    if metric == "ps":
        ylabel = "Parallelism Score"
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
    **Representational geometries** are a computational method for analyzing how 
    populations of neurons encode information about the external environment. By
    projecting the **firing rates** of single neurons in a high-dimensional feature
    space and then examining the properties of the resulting subspace, researchers
    can determine what sort of *coordinated neural activity* enables different behaviors,
    such as successful task performance. This approach reflects the perspective 
    that the functional units of the brain are neuron *ensembles*, rather than individual cells. 

    In contrast with more traditional characterizations of population codes, such as
    sparse coding, representational geometries allow the rich statistical structure
    of neural activity data to be explored directly. For example, instead of pre-processing
    environmental data to determine if individual neurons are responding to suspected
    **latent variables**, or features of the environment that are not directly measured, 
    researchers can apply dimensionality reduction methods to the neural activity space. 
    This accounts for the fact that neurons tend to have **mixed selectivity** rather 
    than being highly specialized. Mixed selectivity is thought to be especially 
    important for *higher cognitive functions* such as abstract reasoning.
    """ 
    st.markdown(intro)

def content_about():
    intro = """
    ##### About this notebook

    This project replicates the key methods and analyses of representational 
    geometries for neuron populations conducted in [1]. In particular, the project
    source code is a custom Python implementation based on descriptions from the 
    original paper and example MatLab scripts included in the open-source dataset.
    
    The app consists of the following sections:
    * **Explore the population code:** a walkthrough of original neural data and processing methods used
    for geometric analyses. DataFrames are embedded for direct interation. 
    * **Visualize neuron state space:** a lower-dimensional visualization of neural
    responses in different brain areas.
    * **Geometric analysis of balanced dichotomies:** attempts to run and plot the three 
    types of the measurements from the original paper (shattering dimensionality, 
    cross-condition generalization performance, and parallelism score). 

    The purpose of this app is to unpack the computational analysis being
    performed in a cutting-edge neuroscience paper to be accessible to undergraduates
    and other non-experts. Thus, the focus of this project is on *exposition of methods* 
    and how they connect to theories of the neural code, rather than justifying the underlying theory that is covered
    by the original paper. This app is also not a full attempt
    to replicate the results of [1] since default parameter values are low in order to keep
    computations feasible (see left sidebar), although the source code
    is available for that purpose.
    """
    st.markdown(intro)
    citation = """
    [1] Courellis, H.S., Minxha, J., Cardenas, A.R. et al. Abstract representations 
    emerge in human hippocampal neurons during inference. Nature 632, 841–849 (2024). 
    [doi.org/10.1038/s41586-024-07799-x](https://doi.org/10.1038/s41586-024-07799-x)
    """
    st.caption(citation)

st.subheader("Introduction")
content_introduction()
content_about()

# ----- Explore Data -----
st.divider()
st.subheader("Explore the population code")

st.markdown("""
Representational geometries aggregate the **firing rates** of individual neurons.
Unlike traditional analyses of firing rates in the literature, however, this approach
does not take the rate values themselves to be encoding meaningful information.
""")

@st.fragment
def content_data_cellular():
    cell_level = """
    ##### Trial-level firing rates for each neuron

    The raw data used for geometric analyses consists of binned spike counts for 
    all neurons across the six brain regions studied in the paper. The following trial-level info
    is available for each neuron. Each set of trials belongs to a single experimental
    session, and each cell has between 180 and 320 trials.
    """
    st.markdown(cell_level)
    cell_idx = st.number_input(
        label=f"Enter an a cell index to view trial data (minimum 0, maximum {MAX_IDX}):",
        min_value=0, max_value=MAX_IDX, value=0
    )
    sample_cell_data = F.get_cell_array(
        st.session_state.neu_data,
        cell_idx
    )
    st.dataframe(sample_cell_data)

@st.fragment
def content_data_population():
    population_level = """
    ##### Constructing a pseudo-population
    
    Neuron recordings from all participants were combined to form a single **pseudo-population.**
    Although this does not reflect a true *population code* from a single brain, 
    the researchers claim this is analagous to how population data is usually constructed:
    by recording from individual neurons and combining them for analysis.

    The final neurons used in computations are then sampled randomly from **balanced
    dichotomies**, or the 35 possible ways that eight task conditions could be 
    split into pairs of four conditions each.
    """
    st.markdown(population_level)
    n_idx = st.number_input(
        label="Enter the number of neurons to sample from:",
        min_value=1,
        max_value=MAX_IDX,
        value=100
    )
    sample_thr = st.number_input(
        label="Set minimum number of samples:",
        min_value=1,
        max_value=None,
        value=15
    )
    population_data = """
    Each row of the dataset corresponds to a single neuron, and each column is
    a specific combination of task conditions. Each cell is an array of firing rates.
    Only correct trials are considered in this part of the analysis. 
    """
    st.markdown(population_data)

    # Choose example indices
    st.session_state.params["curr_idx"] = random.sample(range(0, MAX_IDX), n_idx)
    regressors = F.construct_regressors(st.session_state.params)
    st.dataframe(regressors)

content_data_cellular()
content_data_population()

# ----- Visualize Neuron State Space -----

st.divider()
st.subheader("Visualize neuron state space")
st.markdown("""
Interpretations of representational geometries are supported by both qualitative 
visualizations and quantitative metrics (see following section). In particular,
plotting the neural state space allows researchers to see how the geometric structure
differs between **inference present** trials, where participants successfully generalized
their understanding of a task to a new context, and **inference absent trials**,
where participants failed to answer correctly. 
            
Combined neural data from **inference present** tended to show a greater separation
between the two contexts. Importantly, unlike reward value, **context** is a 
latent variable for each task that is not directly experienced as stimuli. The qualitative
difference in activity between when participants performed a task successfully and
when they did not suggests that performance is related to how well the neuron
population *encodes* the latent context. The researchers primarily saw this difference
in the **hippocampus**.
""")

@st.fragment
def content_state_space():
    plot_area = st.selectbox(
        label="Select a brain region to view:",
        options=['HPC','vmPFC','AMY','dACC','preSMA','VTC'],
        index=0
    )

    intro = """
    Using a dimensionality reduction method called **multi-dimensional scaling**,
    the neural activity space for each brain region can be visualized as a three-dimensional structure.
    In the study, the **geometry of a representation** is defined by the arrangement
    of eight points that represent population responses under different task conditions.
    """
    st.markdown(intro)

    params = st.session_state.params
    params["area_name"] = plot_area
    params["inf_type"] = "inf_pres"
    fig_pres = F.plot_neu_state_space(params)
    params["inf_type"] = "inf_abs"
    fig_abs = F.plot_neu_state_space(params)

    col1, col2 = st.columns(2)
    with col1: 
        st.pyplot(fig_abs)
    with col2:
        st.pyplot(fig_pres)
    caption = """
    Brain area population response during stimulus period in inference present 
    and absent sessions. Pairwise disimilarities between different task conditions
    are visualized in a lower-dimensional space using multidimensional scaling (MDS). Points correspond to stimulus-context combinations (i.e., the 8 task conditions).
    Lines connect the same stimuli across contexts. Context spaces are plotted as
    two planes rather than volumetric surfaces. These figures replicate 2(j) and 3(i) from the original paper.
    """
    st.caption(caption)

content_state_space()


# ----- Geometric Analyses ------

st.divider()
st.subheader("Geometric analysis of balanced dichotomies")

st.markdown("""
A key strength of the representational geometry approach is a straightforward
method of **decoding** information from neural representations. In a typical 
**decoding analysis** for a given **coding scheme**, researchers take the perspective
of a downstream neuron to determine when and what information can be exploited
in a representation. The fields of data science and machine learning have developed
many methods for extracting latent variables from the feature spaces of complicated
datasets, which can be adapted to exploit the neural state space constructed in
this study.

Two metrics that the researchers define to characterize representational geometries,
**shattering dimensionality** and **cross-condition generalization performance**,
reflect how well the neural state space is decoded by a simple linear model
called a **support vector machine**. Importantly, the results of this approach
are not related to whether representational geometries themselves are *biologically
plausible*, meaning that neurons actually have a mechanism to implement this
encoding and decoding strategy. 
            
As with the qualitative plots, the main interest of these metrics is whether
**inference present** and **inference absent**, or successful and unsuccessful,
trials significantly differ.
""")


@st.fragment
def display_single_metric(metric : str):
    if metric not in st.session_state.metrics or st.session_state.params_update:
        st.session_state.metrics[f"{metric}"], st.session_state.metrics[f"{metric}_boot"] = run_single_metric(
            metric=metric,
            n_resample=st.session_state.params["n_resample"],
            n_perm_inner=st.session_state.params["n_perm_inner"],
            n_iter_boot=st.session_state.params["n_iter_boot"],
            n_samples=st.session_state.params["n_samples"]
        )
        st.session_state.params_update = False
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
                data=[data[i][0], data[i][1]],
                null_dist=[data_boot[i][0], data_boot[i][1]],
                y_min=y_min-0.02,
                y_max=y_max+0.02
            )
    
@st.fragment
def content_sd():
    intro = """
    ##### Shattering dimensionality

    **Shattering dimensionality** is defined as the average decoding accuracy across
    all balanced dichotomies. 
    """
    st.markdown(intro)
    display_single_metric("sd")

@st.fragment
def content_ccgp():
    intro = """ 
    ##### Cross-condition generalization performance

    **Cross-condition generalization performance (CCGP)** is a measurement of the 
    extent to which a decoder trained on one set of task conditions can generalize
    to another set. 
    """
    st.markdown(intro)
    display_single_metric("ccgp")

@st.fragment
def content_ps():
    intro = """ 
    ##### Parallelism score 

    The **parallelism score** measures how coding directions for one variable are 
    related to the others using **cosine similarity,** or cosine of the angle between
    two vectors. Cosine similarity values can range from -1, which indicates vectors
    pointing in opposite directions, to 1, which indicates vectors with the same 
    direction. Unlike SD and CCGP, PS does not measure the decodability of a representation.
    """
    st.markdown(intro)
    display_single_metric("ps")

@st.fragment
def select_geometric_analyses():
    analyses = st.multiselect(
        label="Select metric to run:",
        options=["Shattering dimensionality (SD)", "Cross-condition generalization performance (CCGP)", "Parallelism score (PS)"]
    )

    st.caption("""
    The example plots included in this section are for visualization purposes
    only, and should not be seen as true attempts to replicate experimental results.
    All 35 dichotomies are plotted as points. Only named dichotomies are plotted with
    color (see legend on sidebar). Null distributions are represented by gray rectangles.
    """)    

    if "Shattering dimensionality (SD)" in analyses:
        content_sd()
    if "Cross-condition generalization performance (CCGP)" in analyses:
        content_ccgp()
    if "Parallelism score (PS)" in analyses:
        content_ps()


select_geometric_analyses()