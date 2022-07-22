# ValiDyna - An evaluation framework for dynamical systems

## About
This repository contains the code for my MSc thesis research at UofT and EPFL, and for the 2022 ICML SCIS Workshop paper ["Understanding Generalization and Robustess of Learned Representations of Chaotic Dynamical Systems"](https://openreview.net/forum?id=sJg-BRQcLxr).

ValiDyna is the name given to this evaluation framework that can assess the quality of learned representations of dynamical systems.
It allows to write many kinds of experiments in a multi-task setting, and includes the multi-task extension of GRU, LSTM, Transformer and N-BEATS.

## Installation
0. (Pre-requisite) Install `anaconda` or `miniconda`.
1. Clone this git repository and step into it
2. Run the command `conda env create -f environment.yml`
   1. this will create an environment named `chaos`
   2. you can change the environment name in the `environment.yml` file before running the command

## Project structure
- `validyna`: the source code for the project
  - `main.py`: the entry-point to the framework, the only part of the code to be run
  - `models`
    - `multitask_models.py`: the adaptation of N-BEATS, LSTM, GRU and Transformer to a multi-task setting
    - `task_modules.py`: Pytorch-Lightning modules that encapsulate specific tasks with its own training objective and metrics
    - `nbeats.py`: our custom implementation of the [N-BEATS architecture](https://arxiv.org/abs/1905.10437)
  - `metrics.py`: custom metrics
  - `registry.py`: contains all usable model architectures, tasks and metrics, and can be easily extended
  - `data.py`: a helper file with data processing and loading / saving utilities
  - `plot.py`: a helper file with plotting functions
- `experiment_configs`
  - `feature_freeze`: the configs for the experiment "Multi-Task Transfer-Learning" in the paper
  - `few_shot`: the configs for the experiment "One-shot Learning" in the paper
  - `prober`: the configs for a new experiment not appearing in the paper
  - `default.py`: the default configuration used across experiments
- `scripts`
  - `compute_attractor_stats.py`: a script that counts the number of attractors per space dimension
  - `generate_data.py`: a configurable script to generate synthetic chaotic data
- `notebooks`
  - `Plot Results.ipynb`: notebook downloading metrics from wandb and generating plots
- `tests`: directory with the same structure as `validyna` containing tests per file

## Running experiments and writing new ones
Current experiment configurations are in the folder `experiment_configs`.

To run an experiment, execute the command: `python validyna/main.py --cfg=path/to/experiment_config.py`

To add a new experiment config, check the documentation of `run_experiment` and `run_model_training` in `main.py`, as well as `ChunkMultiTaskDataset` in `data.py`.


## Paper and citation
You can find the ValiDyna paper here: https://openreview.net/forum?id=sJg-BRQcLxr

If you use ValiDyna in your work, please cite us using the following bibtex:
```
@inproceedings{streit2022understanding,
    title={Understanding Generalization and Robustess of Learned Representations of Chaotic Dynamical Systems},
    author={Lu{\~a} Streit and Vikram Voleti and Tegan Maharaj},
    booktitle={ICML 2022: Workshop on Spurious Correlations, Invariance and Stability},
    year={2022},
    url={https://openreview.net/forum?id=sJg-BRQcLxr}
}
```
