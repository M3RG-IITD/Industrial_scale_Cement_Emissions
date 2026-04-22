# A Multi-Plant Machine Learning Framework for Emission Prediction, Forecasting, and Control in Cement Manufacturing

## Table of Contents
- [Overview](#overview)
- [System requirements](#system-requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
    - [OS requirements](#os-requirements)
    - [Python dependencies](#python-dependencies)
- [Installation guide](#installation-guide)
- [Reproducing the conda environment](#reproducing-the-conda-environment)
- [Demo](#demo)
- [Queries](#queries)


## Overview
This work establishes a generalizable framework for data-driven emission control in cement production, offering a pathway toward low-emission operation without structural modifications or additional
hardware. We benchmark nine machine learning architectures, and we observe that prediction error varies∼3–5x across plants due to variation in data richness. Incorporating short-term process history nearly triples NOx prediction accuracy, revealing that NOx formation carries substantial process memory, a timescale dependence that is absent in CO and CO2. Further, we develop models that forecast NOx overshoots as early as nine minutes, providing a buffer for operational adjustments. The developed framework controls NOx formation at the source, reducing NH3 consumption in downstream
SNCR. Surrogate model projections estimate a∼34–64% reduction in NOx while preserving clinker quality, corresponding to a reduction of∼290 t NOx/year and∼58,000 USD/year in NH3 savings. Full paper can be found at the following link: [ArXiv reprint](Link_here)

# System requirements

## Hardware requirements
The proposed framework was developed on a system equipped with an Intel® Xeon® Gold 6226R CPU (32 physical cores, 64 threads) and 62.5 GB of RAM. GPU-accelerated models, such as XGBoost, were trained using an NVIDIA RTX A2000 12 GB GPU with driver version 580.65.06 and CUDA version 13.0, utilizing the gpu hist tree method.

## Software requirements

### OS requirements
The framework was trained, validated, and tested on a Linux machine running Ubuntu 22.04 with Python 3.9.12. 

### Python_dependencies
The developed framework requires the following Python libraries.
| Library Type         | Library Name          | Version   |
|---------------------|----------------------|-----------|
| Core                | os                   | built-in  |
| Core                | time                 | built-in  |
| Core                | random               | built-in  |
| Core                | subprocess           | built-in  |
| Core                | threading            | built-in  |
| Core                | resource             | built-in  |
| Core                | statistics           | built-in  |
| Data Handling       | pandas               | 1.4.2     |
| Data Handling       | numpy                | 1.23.5    |
| Machine Learning    | scikit-learn         | 1.2.2     |
| Machine Learning    | xgboost              | 1.7.4     |
| Machine Learning    | optuna               | 4.4.0     |
| Machine Learning    | RandomizedSearchCV   | 1.2.2     |
| Machine Learning    | BayesSearchCV        | 0.10.2    |
| Machine Learning    | scikit-optimize      | 0.10.2    |
| Model Interpretation| shap                 | 0.41.0    |
| Visualization       | matplotlib           | 3.5.1     |
| Visualization       | seaborn              | 0.12.2    |
| Geospatial          | geopandas            | 1.0.1     |
| Geospatial          | shapely              | 2.0.1     |
| GPU                 | pynvml               | 7.352.0   |
| Progress Bar        | tqdm                 | 4.65.0    |

## Installation_guide:
To replicate the environment used for model training and analysis, ensure you have Conda installed. If not, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

## Reproducing the conda environment

### 1. Clone the Repository:
   ```bash
   git clone https://github.com/M3RG-IITD/Industrial-cement-emissions.git
   cd your_repository
   ```
### 2. Create the environment from the .yaml file:
   ```bash
   conda env create -f Conda_Environment/environment.yaml
   ```
### 3. Activate the environment:
   ```bash
   conda activate base
   ```
### 4. Check if the environment is present:
   ```bash
   conda list
   ```
## Demo

## Queries
For any queries on the work, please email cez218290@iitd.ac.in.

