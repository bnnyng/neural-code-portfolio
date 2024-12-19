# Neural Code Final Portfolio

This respository is a Python re-implementation of Courellis et al. (2024), "[Abstract representations emerge in human hippocampal neurons during inference](https://www.nature.com/articles/s41586-024-07799-x)." Original analyses were performed in MatLab.

Data files:
* `beh_data.json`: behavioral data organized by recording session
* `task_data.json`: information about trial-level task variables
* `neu_data.json`: trial-level recording data from individual neurons

Code:
* `analysis.ipynb`: notebook with detailed explanations of data pre-processing and examples of how to use the methods
* `helper_functions.py`: all methods for data processing, geometric analyses, and plotting

The corresponding [StreamLit app](https://neural-code-portfolio.streamlit.app/) has examples cellular-level data used to train the linear decoders, as well as plots for neural state space and toy results for geometric analyses.
