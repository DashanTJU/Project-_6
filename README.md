# Project-_6


PV power production and solar irradiance forecasting with hybrid deep learning and DALF

Standalone recurrent networks (RNN, GRU, LSTM), hybrid CNN-recurrent networks (HCRM, HCGM, HCLM) and three fusion methods for two PV plants:

* DALF - Data-Adaptive Linear Fusion: daily re-estimated, non-negative weights that sum to one and minimise the MAE of the preceding 30 completed days;
* SCAF - Static Coefficient Aggregation Fusion: one weight vector fitted on the validation period and frozen for the test period;
* UEBF - Uniform Ensemble Blending: equal weights (1/3 each).

Task A forecasts 24 hourly power values and evaluates them as **daily energy**; Task B forecasts 24 hourly **irradiance** values and evaluates them hourly. The two tasks are fully independent: own cleaning, scalers, models, fusion weights and results.

Methodology implemented

Cleaning (per plant and target
Split, scaling, windows
Models and training
Temporal resolution of fusion 
DALF update protocol
SCAF
Metrics
Statistical tests
Ablation and robustness

Python
tensorflow
numpy
pandas
scikit-learn
scipy
matplotlib
openpyxl
psutil
pytest


