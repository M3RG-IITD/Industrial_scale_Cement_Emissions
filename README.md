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
- [Code functionality: pseudocode for NOx control framework](#code-functionality-pseudocode-for-nox-control-framework)
- [Demo](#demo)
   - [Steps to Run](#steps-to-run)
   - [Expected Output](#expected-output)
- [Queries](#queries)


## Overview
This work establishes a generalizable framework for data-driven emission control in cement production, offering a pathway toward low-emission operation without structural modifications or additional
hardware. We benchmark nine machine learning architectures, and we observe that prediction error varies ∼3–5x across plants due to variation in data richness. Incorporating short-term process history nearly triples NO<sub>x</sub> prediction accuracy, revealing that NO<sub>x</sub> formation carries substantial process memory, a timescale dependence that is absent in CO and CO<sub>2</sub>. Further, we develop models that forecast NOx overshoots as early as nine minutes, providing a buffer for operational adjustments. The developed framework controls NOx formation at the source, reducing NH<sub>3</sub> consumption in downstream SNCR. Surrogate model projections estimate a ∼34–64% reduction in NO<sub>x</sub> while preserving clinker quality, corresponding to a reduction of ∼290 t NO<sub>x</sub>/year and ∼58,000 USD/year in NH3 savings. Full paper can be found at the following link: [ArXiv reprint](Link_here)

# System requirements

## Hardware requirements
The proposed framework was developed on a system equipped with an Intel® Xeon® Gold 6226R CPU (32 physical cores, 64 threads) and 62.5 GB of RAM. GPU-accelerated models, such as XGBoost, were trained using an NVIDIA RTX A2000 12 GB GPU with driver version 580.65.06 and CUDA version 13.0, utilizing the gpu hist tree method.

## Software requirements

### OS requirements
The framework was trained, validated, and tested on a Linux machine running Ubuntu 22.04 with Python 3.9.12. 

### Python dependencies
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

## Installation guide:
To replicate the environment used for model training and analysis, ensure you have Conda installed. If not, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (takes upto 5 mins) or [Anaconda](https://www.anaconda.com/) (takes up to 20 mins).

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
Creating the environment from the .yaml file can take up to 30-45 mins.
### 3. Activate the environment:
   ```bash
   conda activate base
   ```
### 4. Check if the environment is present:
   ```bash
   conda list
   ```
Once the conda environment is set up, it can be used to run the code and demo, provided the specified hardware and software requirements are met.
## Code functionality: pseudocode for NOx control framework

**Input:**
- Initial plant state X₀ (NOx, clinker flow, free lime, process variables)  
- Trained ML surrogate models:  
  - M_NOx → predicts NOx emissions  
  - M_clinker → predicts clinker flow rate  
  - M_fCaO → predicts free lime (f-CaO)  
- Historical plant data (for correlation and realism constraints)  

**Output:**
- Optimized decision variables (DVs)  
- Minimized NOx value  
- Validated control recommendations  

---

**1. Define Decision Variables (DVs):**  
Select controllable process parameters (e.g., air flows, fuel rates, fan speed, kiln pressure)

**2. Fix Non-controllable Inputs:**  
Keep raw meal flow rate and chemistry constant

**3. Define Constraints:**  
- Each DV can vary within ±5% of its initial value  
- Total fuel consumption must not increase  

**4. Initialize Genetic Algorithm (GA):**  
- Generate initial population within bounded DV space  
- Lock fixed variables (e.g., raw meal flow)  

---

**5. Optimization Loop (GA):**  

For each generation:  

- For each candidate solution x in the population:  
  - Check constraints; if violated, assign penalty and skip  
  - Predict NOx using surrogate model: NOx_pred = M_NOx(x)  
  - Compute correlation penalty (Pcorr) using deviation from historical relationships  
  - Compute realism penalty (Preal) using deviation from historical data manifold  
  - Predict clinker flow: Clinker_pred = M_clinker(x)  
  - If clinker deviation exceeds limit, add penalty  
  - Compute objective:  
    J(x) = NOx_pred + Wcorr * Pcorr + Wreal * Preal  

- Apply GA operations:  
  - Selection (tournament selection)  
  - Crossover (one-point crossover)  
  - Mutation (bounded perturbations)  

- Form next generation population  

---

**6. Select Optimal Solution:**  
Choose the solution x* with the minimum objective value J(x)

---

**7. KPI Validation:**  
- Predict clinker flow and free lime using surrogate models  
- Check constraints:  
  - Clinker flow variation ≤ 0.5%  
  - 0.5 ≤ fCaO ≤ 1.5  
- Accept solution if all constraints are satisfied; otherwise reject  

---

**8. Output Results:**  
- Optimized decision variables (x*)  
- Reduced NOx level  
- KPI validation status  

---

**9. Optional Practical Validation:**  
Compare optimized solution with historical plant states using similarity metrics  

## Demo
After setting up the conda environment and ensuring the system meets the specified requirements, follow the steps below to run the demo:

### Steps to Run
1. Navigate to the `Demo` folder in this repository  
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
3. Open the Demo.ipynb file
4. Run all cells sequentially to execute the NO<sub>x</sub> control framework
### Expected Output
Successful execution of the Demo notebook (take ~10 mins) will load and preprocess the input process data, display model predictions for NO<sub>x</sub> emissions, and provide insights into emission control behavior.
## Queries
For any queries on the work, please email cez218290@iitd.ac.in.

