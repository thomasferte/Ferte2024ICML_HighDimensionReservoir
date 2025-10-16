# High-Dimensional Reservoir Computing for Short Time Series Forecasting

[![DOI](https://zenodo.org/badge/735935333.svg)](https://doi.org/10.5281/zenodo.17295478)

[![ICML
2024](https://img.shields.io/badge/ICML-2024-blue.svg)](https://proceedings.mlr.press/v235/ferte24a.html)

This repository contains the official implementation of the ICML 2024
paper:

> **Reservoir Computing for Short High-Dimensional Time Series: an
> Application to SARS-CoV-2 Hospitalization Forecast**\
> *Thomas Ferté, Dan Dutartre, Boris P. Hejblum, Romain Griffier,
> Vianney Jouhet, Rodolphe Thiébaut, Pierrick Legrand, Xavier Hinaut*\
> [Proceedings of Machine Learning Research, Vol. 235, ICML
> 2024](https://proceedings.mlr.press/v235/ferte24a.html)

------------------------------------------------------------------------

## 🧠 Overview

This repository implements a **Reservoir Computing (RC)** approach
enhanced with a **Genetic Algorithm (GA)** for feature selection,
designed for **short, high-dimensional time series forecasting**.\
The method improves forecasting performance and stability in
data-limited settings, compared with baseline models such as LSTM,
Transformer, ElasticNet, and XGBoost.

An application to **SARS-CoV-2 hospitalization forecasts** is presented
in the ICML 2024 paper.

------------------------------------------------------------------------

## 📂 Repository Structure

```         
Ferte2024ICML_HighDimensionReservoir/
├── data/                         # Datasets
├── genetic_algorithm/            # Genetic Algorithm for feature selection
├── pre_compute_smoothing/        # Preprocessing and smoothing utilities
├── script/                       # Auxiliary and orchestration scripts
├── test_algorithm/               # Testing and evaluation scripts
├── train_test_api/               # Unified API for training/testing pipelines
├── esn_dataset.py                # Dataset class and input matrix building for reservoir computing
├── trans_utils.py                # Time-series transformation utilities
├── main_csv_evaluate.py          # Evaluation and metrics collection
├── main_csv_test.py              # Test orchestration
├── measure_time.py               # Timing and complexity measurement
├── lsmt_for_time_computtation.py # Runtime and comparison with LSTM baselines
├── 2000_reservoir.py             # Example experiment: 2000-unit reservoir
├── 2000_20reservoir.py           # Experiment variant with 20 reservoirs
├── xgb_pred_RS*.slurm            # XGBoost baseline experiments
├── enet_pred_RS*.slurm           # Elastic-Net baseline experiments
├── GeneticSingleIS_*.slurm       # Reservoir Computing experiments
├── prophet.slurm                 # Prophet baseline
├── transformers_time_computtation.ipynb  # Notebook for profiling / transformer timing
├── *.slurm                       # Slurm job submission scripts for HPC clusters
├── high_dimension_reservoir.Rproj # R project file for supplementary analysis
└── .gitignore
```

## ⚙️ Installation

### **Clone the repository**

```{bash}
 git clone https://github.com/thomasferte/Ferte2024ICML_HighDimensionReservoir.git
```

## 🚀 Running Experiments

1.  Data Preparation

Use `script/precompute_smoothin.R` to preprocess `data/df_obuscated.rds`. This 
script aims to read the dataset, perform smoothing and derivative computation. 
Then it saves multiple csv files that contain the training data smoothed up to 
a given date. Those csv files will then be called for training and testing.

2.  Training and Testing

Experiments are designed to run on an HPC system using Slurm job
submissions. Each Slurm script has a corresponding "test" version for
evaluation on test set.

-    **Training:** Submit the main Slurm script to train the
    high-dimensional reservoir models.

-   **Testing:** Use the "test" script to evaluate the trained model.

Both training and testing scripts internally call:

-   `main_csv_evaluate.py` – handles model training. It has three steps. First 
it determines the setting from experiment name. Then it will determine the 
optimal hyperparameter sets each month and save them.

-   `main_csv_test.py` – handles model evaluation. It loads the hyperparameters 
sets found by main_csv_evaluate and test the algorithm with those

-   `main_csv_test.py` – handles model evaluation. It loads the hyperparameters 
sets found by main_csv_evaluate and test the algorithm with those

-   `read_test_files_after_csv_evaluate.py` called by 
`csv_compile_evaluation.slurm` then aggregate all the files for easier 
reporting.

3.  Reporting

Reporting of the experiments is done in the script folder by
`result_high_dim_rc.qmd` file.

## 📚 Citation

If you use this work, please cite:

```         
@InProceedings{pmlr-v235-ferte24a,
  title = 	 {Reservoir Computing for Short High-Dimensional Time Series: an Application to {SARS}-{C}o{V}-2 Hospitalization Forecast},
  author =       {Fert\'{e}, Thomas and Dutartre, Dan and Hejblum, Boris P and Griffier, Romain and Jouhet, Vianney and Thi\'{e}baut, Rodolphe and Legrand, Pierrick and Hinaut, Xavier},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {13570--13591},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/ferte24a/ferte24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/ferte24a.html},
  abstract = 	 {In this work, we aimed at forecasting the number of SARS-CoV-2 hospitalized patients at 14 days to help anticipate the bed requirements of a large scale hospital using public data and electronic health records data. Previous attempts led to mitigated performance in this high-dimension setting; we introduce a novel approach to time series forecasting by providing an alternative to conventional methods to deal with high number of potential features of interest (409 predictors). We integrate Reservoir Computing (RC) with feature selection using a genetic algorithm (GA) to gather optimal non-linear combinations of inputs to improve prediction in sample-efficient context. We illustrate that the RC-GA combination exhibits excellent performance in forecasting SARS-CoV-2 hospitalizations. This approach outperformed the use of RC alone and other conventional methods: LSTM, Transformers, Elastic-Net, XGBoost. Notably, this work marks the pioneering use of RC (along with GA) in the realm of short and high-dimensional time series, positioning it as a competitive and innovative approach in comparison to standard methods.}
}
```
