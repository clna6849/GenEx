# Stratified TES MLP Surrogate Model

This repository provides an inference-only interface for a trained multilayer perceptron (MLP) surrogate model for predicting the transient vertical temperature distribution and thermodynamic performance of water-based stratified thermal energy storage (TES) systems during stand-by operation.

The surrogate model predicts the time-resolved vertical temperature profile of a stratified TES system within the investigated design space. From the predicted temperature profile, the script also calculates the absolute energy content, absolute exergy content, normalized energy content, and normalized exergy content at a user-defined stand-by time.

No CFD training data, preprocessing routines, hyperparameter optimization, cross-validation code, or training procedure are included in this repository. The repository is intended for prediction-only use based on the trained model cache `final_prediction_cache.pt`.

## Repository structure

```text
GenEx/
├── models/
│   └── final_prediction_cache.pt
├── tes_mlp_prediction.py
├── requirements.txt
└── README.md
