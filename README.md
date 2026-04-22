# A Multi-Plant Machine Learning Framework for Emission Prediction, Forecasting, and Control in Cement Manufacturing
Full paper can be found at the following link: [ArXiv reprint](Link_here)

## System requirements
All computations were conducted on a Linux machine running Ubuntu 22.04 with Python 3.9.12. The system was equipped with an Intel® Xeon® Gold 6226R CPU (32 physical cores, 64 threads) and 62.5 GB of RAM. GPU-accelerated models, such as XGBoost, were trained using an NVIDIA RTX A2000 12 GB GPU with driver version 580.65.06 and CUDA version 13.0, utilizing the gpu hist tree method.

## Computational Environment (Python libraries)
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

## Instructions on installing the git repo and reproducing the Environment

To replicate the environment used for model training and analysis, follow these steps:

### 1. Install Conda:
   Ensure you have Conda installed. If not, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

### 2. Clone the Repository:
   ```bash
   git clone https://github.com/M3RG-IITD/Industrial-cement-emissions.git
   cd your_repository
   ```
### 3. Create the environment from the .yaml file:
   ```bash
   conda env create -f Conda_Environment/environment.yaml
   ```
### 4. Activate the environment:
   ```bash
   conda activate base
   ```
### 5. Check if the environment is present:
   ```bash
   conda list
   ```

For any queries on the work, please email cez218290@iitd.ac.in.

