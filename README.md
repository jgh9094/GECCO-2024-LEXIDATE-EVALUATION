# Lexidate validation evaluation strategy

## Overview

Link to all supplementary material: [here](https://jgh9094.github.io/GECCO-2024-LEXIDATE-EVALUATION/Bookdown/Pages/).
All data is available on the Open Science Framework [here](https://osf.io/mnzjg/)

### Abstract

> Automated machine learning (AutoML) aims to streamline the process of finding effective machine learning pipelines by automating model training, evaluation, and selection.
As part of that process, evaluating a pipeline consists of using an evaluation strategy to estimate its generalizability.
Traditional evaluation strategies, like cross-validation, generate one value that summarizes the quality of a pipeline's predictions.
This single value, however, may not be conducive to evolving effective pipelines.
Here, we present Lexicase-based Validation (lexidate), an evaluation strategy that uses multiple prediction values instead of an aggregated value.
Lexidate splits data into a learning set and a selection set.
Pipelines are trained on the learning set and make predictions on the selection set; predictions are graded for correctness.
These graded predictions are used by lexicase selection to identify pipelines to continue evolving.
We test the effectiveness of lexidate within the Tree-based Pipeline Optimization Tool 2 (TPOT2) package on six OpenML classification tasks.
In one configuration of lexidate, we found no difference in the accuracy of the final models returned from TPOT2 on most of the tasks when compared to 10-fold cross-validation.
For all three lexidate configurations studied here, similar or less complex final pipelines were found when compared to 10-fold cross-validation.

## Repository guide

Datasets used in the experiments. The `Task ID' refers to the identifier used to extract the dataset from OpenML. The other columns denote the number of rows, columns, and classes for each dataset.

- `Data-Tools/`: all scripts related to data checking, collecting, and visualizing
  - `Check/`: scripts for checking data
  - `Collect/`: scripts for collecting data
  - `Visualize/`: scripts for making plots
- `Experiments/`: all scripts to run experiments on HPC
  - `Base/`: runs related to 10-fold cross-validation with TPOT2
    - `HPC/`: scripts for HPC
  - `Splits/`: runs related to lexidate validation with TPOT2
    - `HPC/`: scripts for HPC
- `Source/`: contains all Python scripts to run experiments.
- `tpot2-base`: TPOT2 implementation for 10-fold cross-validation runs
- `tpot2-sol-obj`: TPOT2 implementation for lexidate validation implementation runs


## OpenML classification tasks

| Name                    | Task ID | Rows | Columns | Classes |
|-------------------------|---------|------|---------|---------|
| australian              | 167104  | 690  | 15      | 2       |
| blood-transfusion...    | 167184  | 748  | 5       | 2       |
| vehicle                 | 167168  | 846  | 19      | 4       |
| credit-g                | 167161  | 1000 | 21      | 2       |
| cnae-9                  | 167185  | 1080 | 857     | 9       |
| car                     | 189905  | 1728 | 7       | 4       |

## TPOT2 configurations

| Parameter                | Values        |
|--------------------------|---------------|
| Population size          | 48            |
| Number of generations    | 200           |
| Mutation operators       | See below     |
| Mutation rate            | 100%          |
| Number of runs per condition | 40        |


## TPOT2 muation operators

- mutate_hyperparameters: Vary the encoded hyperparameters of a random node.
- mutate_replace_node: Assign a new ML method and hyperparameters to a random node.
- mutate_remove_node: Remove a random leaf or inner node, and connect all its children nodes to all its parent nodes.
- mutate_remove_edge: Remove an outgoing edge of a random node with more than one outgoing edge.
- mutate_add_edge: Add an edge between two random nodes.
- mutate_insert_leaf: Connect a new leaf node to a random node.