# Riemannian Optimization for Fair Spectral Clustering

## Overview

This repository contains the implementation of R-FairSC, a scalable fair spectral clustering algorithm based on Riemannian manifold optimization.

R-FairSC formulates fair spectral clustering as a constrained optimization problem on a Riemannian manifold and develops a Riemannian ADMM algorithm with a variable-splitting strategy to efficiently solve the resulting subproblems. By avoiding computationally expensive eigendecompositions of the graph Laplacian, R-FairSC improves scalability while maintaining high clustering quality and fairness on large synthetic and real-world graphs.

## Installation

We recommend using the conda virtual environment:

```bash
$ conda env create -f environment.yml
$ conda activate r-fairsc
```

## Dataset

We evaluate the methods on four real-world datasets:

- Credit
- Bail
- Bank
- Diabetes

We also provide experiments on synthetic graphs generated using a stochastic block model (SBM).

The dataset files should be placed under the `data/` directory.

## Running the code

Follow the commands below to run the experiments.

### R-FairSC

Run R-FairSC on the real-world datasets:

```bash
$ python -m experiments.run_credit
$ python -m experiments.run_bail
$ python -m experiments.run_bank
$ python -m experiments.run_diabetes
```

Run R-FairSC on the synthetic SBM dataset:

```bash
$ python -m experiments.run_sbm
```
### Baselines

### A-FairSC

```bash
$ python -m experiments.run_credit_afairsc
$ python -m experiments.run_bail_afairsc
$ python -m experiments.run_bank_afairsc
$ python -m experiments.run_diabetes_afairsc
$ python -m experiments.run_sbm_afairsc
```

### sFairSC

```bash
$ python -m experiments.run_credit_sfairsc
$ python -m experiments.run_bail_sfairsc
$ python -m experiments.run_bank_sfairsc
$ python -m experiments.run_diabetes_sfairsc
```

### Standard Spectral Clustering

```bash
$ python -m experiments.run_credit_sc
$ python -m experiments.run_bail_sc
$ python -m experiments.run_bank_sc
$ python -m experiments.run_diabetes_sc
```

Experimental results are saved under the `results/` directory.

## Publication

Minh Phu Vuong, Jinyoung Lee, Young-Ju Lee, and Chul-Ho Lee. *Riemannian Optimization for Fair Spectral Clustering*. In *Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*.

## Citation

If you use R-FairSC in your research, please cite:

```bibtex
@inproceedings{vuong2026rfairsc,
  title     = {Riemannian Optimization for Fair Spectral Clustering},
  author    = {Vuong, Minh Phu and Lee, Jinyoung and Lee, Young-Ju and Lee, Chul-Ho},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
