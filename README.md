# Lexidate validation evaluation strategy

## Overview

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
- `tpot2-sol-obj`: TPOT2 implemenation for lexidate validation implementation runs
