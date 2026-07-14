# Stratified TES MLP Surrogate Model

This repository provides a multilayer perceptron (MLP) surrogate model for predicting the transient vertical temperature distribution and thermodynamic performance of water-based stratified thermal energy storage (TES) systems during stand-by operation. The model predicts the time-resolved vertical temperature profile of a stratified TES system within the investigated design space. From the predicted temperature profile, the model also calculates the absolute energy content, absolute exergy content, normalized energy content, and normalized exergy content at a user-defined stand-by time. For further information, please find the corresponding paper here {insert URL}. The repository is intended for prediction-only use based on the trained model cache `final_prediction_cache.pt`.

## Repository structure

```
GenEx/
├── models/
│   └── final_prediction_cache.pt
├── tes_mlp_prediction.py
├── requirements.txt
└── README.md
```
## Installation

Clone the GitHub repository:

```text
git clone https://github.com/clna6849/GenEx.git
cd GenEx
```

Create a new python environment using Python 3.11.x:

```text
conda create -n genex python=3.11
conda activate genex
```

Install the requirements:

```
pip install -r requirements.txt
```
The model was tested for Python 3.11.14, NumPy 2.4.0, PyTorch 2.9.1 and scikit-learn 1.8.0. After successful installation, the script `tes_mlp_prediction.py` can be used directly from the repository folder, for example in Spyder, VS Code, Jupyter, or any other Python environment, as long as the working directory is set to the folder containing `tes_mlp_prediction.py` and the `models` directory.

## Basic usage

```python
from tes_mlp_prediction import load_surrogate, predict_tes

model = load_surrogate("models/final_prediction_cache.pt", device="cpu")

result = predict_tes(
    model,
    H=2.0,
    aspect_ratio=1.0,
    k0=55.0,
    wall_thickness_mm=5.5,
    tau_h=1200.0,
    time_h=24.0,
)

print(result["energy"]["kWh"])
print(result["exergy"]["kWh"])
print(result["energy"]["normalized"])
print(result["exergy"]["normalized"])
print(result["temperature_K"])
```

The returned temperature profile corresponds to the vertical profile at the requested stand-by time.

## Input parameters

The model uses five input parameters:

| Parameter | Description | Physical range |
|---|---|---:|
| `H` | Storage height in m | 1 m to 3 m |
| `aspect_ratio` | Diameter-to-height ratio | 0.5 to 1.5 |
| `k0` | Initial sigmoid slope parameter describing the initial degree of stratification | 10 to 100 |
| `wall_thickness_mm` | Storage tank wall thickness in mm | Geometry-dependent |
| `tau_h` | Storage time constant in h, representing the thermal insulation level | Geometry-dependent |

The admissible ranges of `wall_thickness_mm` and `tau_h` depend on the storage geometry, becuase they are scaled with the storage height and aspect ratio according to the original design space. The scaling can be seen in the function `doe_bounds` in the main prediction script. Each parameter can be provided either as a physical value, or as a relative value in the interval `[0,1]`. For each parameter, only one represntation may be given. 

### Full physical input

```python
result = predict_tes(
    model,
    H=2.0,
    aspect_ratio=1.0,
    k0=55.0,
    wall_thickness_mm=5.5,
    tau_h=1200.0,
    time_h=24.0,
)
```

### Full relative input

```python
result = predict_tes(
    model,
    H_rel=0.5,
    aspect_ratio_rel=0.5,
    k0_rel=0.5,
    d_rel=0.5,
    tau_rel=0.5,
    time_h=24.0,
)
```

## Output structure

The function `predict_tes(...) returns a dictionary with the following output fields:

| Output field | Description |
|---|:---|
| result["time_h"] | Desired point in stand-by time |
| result["input_relative"] | relative input parameter values |
| result["input_physical"] | physical input parmaeter values |
| result["geometry"] | contains the diameter, radius, cross section area, outer surface area, volume, volume-to-surface ratio |
| result["heat_transfer"] | contains the outer heat transfer coefficient, ambient temperature, reference values of density and specific heat capacity, storage time constant |
| result["z_rel"] | relative vertical coordinate |
| result["z_abs_m"] | absolute vertical coordinate |
| result["temperature_K"] | averaged vertical temperature data |
| result["energy"] | contains the calculated absolute and normalized energy |
| result["exergy"] | contains the calculated absolute and normalized exergy |
| result["mass_kg"] | total mass of the storage medium |

## Returning the full time series

By default, `predict_tes(...)` returns the results at the requested stand-by time only. To return the full predicted time series, use:

```python
result = predict_tes(
    model,
    H=2.0,
    aspect_ratio=1.0,
    k0=55.0,
    wall_thickness_mm=5.5,
    tau_h=1200.0,
    time_h=24.0,
    return_time_series=True,
)
```
This adds:

```python
result["time_series"]
```
with the following entries:

```python
result["time_series"]["time_s"]
result["time_series"]["time_h"]
result["time_series"]["temperature_K"]
result["time_series"]["energy_J"]
result["time_series"]["energy_MJ"]
result["time_series"]["energy_kWh"]
result["time_series"]["energy_norm"]
result["time_series"]["exergy_J"]
result["time_series"]["exergy_MJ"]
result["time_series"]["exergy_kWh"]
result["time_series"]["exergy_norm"]
result["time_series"]["mass_kg"]
```
The full time series can be used for plotting, post-processing, or user-defined optimization studies.

## Prediction cache

The cache was created with torch.save(...) and contains all objects required for the prediction. It does not contain the CFD raw data, the full training data set, the preprocessing workflow, the hyperparameter optimization, or the cross-validation routine. The cache is loaded by:

```python
from tes_mlp_prediction import load_surrogate

model = load_surrogate("models/final_prediction_cache.pt")
```

It has the following main entries:

| Entry | Description |
|---|:---|
| "cfg" | contains the configuration required for reconstructing the surrogate model and the vertical temperature profiles |
| "t" | time vector of the surrogate model |
| "model_state_dict" | contains the trainey PyTorch weights and biases of the MLP |
| "pcas" | contains the fitted scikit-learn PCA objects used for reconstructing the time-dependent sigmoid parameter curves |
| "x_scaler" | Standard Scaler used to standardize the MLP input parameters before prediction |
| "y_scaler" | Standard Scaler used to for the MLP output |

The raw MLP output corresponds to standardized PCA coefficients. Before inverse PCA reconstruction, the coefficients must be transformed back to their original scale:

```python
Y = payload["y_scaler"].inverse_transform(Y_scaled)
```

By using the provided prediction cache, users can also build their own constrained optimization problem based on individual cost functions, for example to obtain the optimal storage geometry for a given required exergy efficiency.
