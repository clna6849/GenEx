# Stratified TES MLP Surrogate Model

This repository provides a multilayer perceptron (MLP) surrogate model for predicting the transient vertical temperature distribution and thermodynamic performance of water-based stratified thermal energy storage (TES) systems during stand-by operation. The model predicts the time-resolved vertical temperature profile of a stratified TES system within the investigated design space. From the predicted temperature profile, the model also calculates the absolute energy content, absolute exergy content, normalized energy content, and normalized exergy content at a user-defined stand-by time.

The repository is intended for prediction-only use based on the trained model cache `final_prediction_cache.pt`.

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

Install the required Python packages using Python 3.11.x or newer:

```
pip install -r requirements.txt
```

The model was tested for Python 3.11.14, NumPy 2.4.0, PyTorch 2.9.1 and scikit-learn 1.8.0.
