"""
# =============================================================================
# tes_mlp_prediction.py
#
# Author: Clemens Naumann
# Description: Prediction interface for the trained MLP surrogate
#              model of a stratified thermal energy storage system.
#
# Note: Parts of this code were developed with assistance from an AI language
# model (OpenAI ChatGPT). The scientific concept, model training, validation,
# and final verification remain the responsibility of the author.
# =============================================================================
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Model architecture
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """Multilayer perceptron architecture used by the trained surrogate."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TESSurrogate:
    """Container for the loaded inference-only surrogate model."""

    model: MLP
    pcas: Dict[str, PCA]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    t_s: np.ndarray
    cfg: Dict[str, Any]
    device: torch.device


# -----------------------------------------------------------------------------
# Parameter transformation and geometry
# -----------------------------------------------------------------------------


PUBLIC_RELATIVE_PARAMETER_ORDER = [
    "H_rel",
    "aspect_ratio_rel",
    "k0_rel",
    "d_rel",
    "tau_rel",
]

MLP_INPUT_ORDER = [
    "H_rel",
    "aspect_ratio_rel",
    "k_log_rel_internal",
    "d_rel",
    "tau_rel",
]


def doe_bounds(H: float, aspect_ratio: float) -> tuple[float, float, float, float]:
    """
    Return admissible wall-thickness and storage-time-constant bounds.

    Parameters
    ----------
    H : float
        Storage height in m.
    aspect_ratio : float
        Diameter-to-height ratio D/H.

    Returns
    -------
    d_min_mm, d_max_mm, tau_min_h, tau_max_h : tuple of float
        Geometry-dependent bounds used in the original design of experiments.
    """

    H_min, H_max = 1.0, 3.0
    aspect_ratio_min, aspect_ratio_max = 0.5, 1.5

    R = aspect_ratio * H / 2.0
    R_small = aspect_ratio_min * H_min / 2.0
    R_large = aspect_ratio_max * H_max / 2.0

    S = H * 2.0 * R
    S_small = H_min * 2.0 * R_small
    S_large = H_max * 2.0 * R_large
    wS = np.clip((S - S_small) / (S_large - S_small), 0.0, 1.0)

    d_min_mm = 1000.0 * (0.002 + wS * (0.008 - 0.002))
    d_max_mm = 1000.0 * (0.005 + wS * (0.015 - 0.005))

    V = np.pi * R**2 * H
    A = 2.0 * np.pi * R * H + 2.0 * np.pi * R**2

    V_small = np.pi * R_small**2 * H_min
    A_small = 2.0 * np.pi * R_small * H_min + 2.0 * np.pi * R_small**2

    V_large = np.pi * R_large**2 * H_max
    A_large = 2.0 * np.pi * R_large * H_max + 2.0 * np.pi * R_large**2

    wL = np.clip((V / A - V_small / A_small) / (V_large / A_large - V_small / A_small), 0.0, 1.0)

    tau_min_h = 100.0 + wL * (2000.0 - 100.0)
    tau_max_h = 500.0 + wL * (3000.0 - 500.0)

    return float(d_min_mm), float(d_max_mm), float(tau_min_h), float(tau_max_h)


def storage_geometry(H: float, aspect_ratio: float) -> Dict[str, float]:
    """Calculate cylindrical storage geometry from height and aspect ratio."""

    diameter_m = float(aspect_ratio * H)
    radius_m = diameter_m / 2.0
    cross_section_area_m2 = float(np.pi * radius_m**2)
    volume_m3 = float(cross_section_area_m2 * H)

    # Outer surface area of a closed cylinder: side wall + top + bottom.
    outer_surface_area_m2 = float(2.0 * np.pi * radius_m * H + 2.0 * np.pi * radius_m**2)
    volume_to_surface_ratio_m = float(volume_m3 / outer_surface_area_m2)

    return {
        "diameter_m": diameter_m,
        "radius_m": radius_m,
        "cross_section_area_m2": cross_section_area_m2,
        "outer_surface_area_m2": outer_surface_area_m2,
        "volume_m3": volume_m3,
        "volume_to_surface_ratio_m": volume_to_surface_ratio_m,
    }


def k0_to_k0_rel(k0: float) -> float:
    """Convert physical initial sigmoid slope k0 to public linear relative k0_rel."""
    return float((float(k0) - 10.0) / (100.0 - 10.0))


def k0_rel_to_k0(k0_rel: float) -> float:
    """Convert public linear relative k0_rel to physical initial sigmoid slope k0."""
    return float(10.0 + float(k0_rel) * (100.0 - 10.0))


def k0_to_internal_log_relative(k0: float) -> float:
    """
    Convert physical k0 to the logarithmic relative value used internally by the MLP.

    This value is intentionally not exposed in the public prediction interface.
    """
    return float((np.log(float(k0)) - np.log(10.0)) / (np.log(100.0) - np.log(10.0)))


def public_relative_to_mlp_input(X_rel_public: np.ndarray) -> np.ndarray:
    """
    Convert public relative input to the internal MLP input.

    Public relative order:
        [H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel]

    Internal MLP order:
        [H_rel, aspect_ratio_rel, log-scaled k0, d_rel, tau_rel]
    """

    X_rel_public = np.atleast_2d(np.asarray(X_rel_public, dtype=float))
    check_public_relative_bounds(X_rel_public)

    X_mlp = X_rel_public.copy()
    k0 = 10.0 + X_rel_public[:, 2] * (100.0 - 10.0)
    X_mlp[:, 2] = (np.log(k0) - np.log(10.0)) / (np.log(100.0) - np.log(10.0))
    return X_mlp


def physical_to_relative(params: np.ndarray) -> np.ndarray:
    """
    Convert physical parameters to the public relative input space.

    Physical parameter order:
        [H, aspect_ratio, k0, wall_thickness_mm, tau_h]

    Public relative output order:
        [H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel]
    """

    params = np.atleast_2d(np.asarray(params, dtype=float))
    X_rel = np.zeros((params.shape[0], 5), dtype=float)

    for i, (H, aspect_ratio, k0, wall_thickness_mm, tau_h) in enumerate(params):
        d_min_mm, d_max_mm, tau_min_h, tau_max_h = doe_bounds(H, aspect_ratio)

        X_rel[i, 0] = (H - 1.0) / (3.0 - 1.0)
        X_rel[i, 1] = (aspect_ratio - 0.5) / (1.5 - 0.5)
        X_rel[i, 2] = k0_to_k0_rel(k0)
        X_rel[i, 3] = (wall_thickness_mm - d_min_mm) / (d_max_mm - d_min_mm)
        X_rel[i, 4] = (tau_h - tau_min_h) / (tau_max_h - tau_min_h)

    return X_rel


def relative_to_physical(X_rel: np.ndarray) -> np.ndarray:
    """
    Convert public relative input values to physical parameters.

    Public relative input order:
        [H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel]

    Physical output order:
        [H, aspect_ratio, k0, wall_thickness_mm, tau_h]
    """

    X_rel = np.atleast_2d(np.asarray(X_rel, dtype=float))
    check_public_relative_bounds(X_rel)

    params = np.zeros((X_rel.shape[0], 5), dtype=float)

    for i, (H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel) in enumerate(X_rel):
        H = 1.0 + H_rel * (3.0 - 1.0)
        aspect_ratio = 0.5 + aspect_ratio_rel * (1.5 - 0.5)
        k0 = k0_rel_to_k0(k0_rel)

        d_min_mm, d_max_mm, tau_min_h, tau_max_h = doe_bounds(H, aspect_ratio)
        wall_thickness_mm = d_min_mm + d_rel * (d_max_mm - d_min_mm)
        tau_h = tau_min_h + tau_rel * (tau_max_h - tau_min_h)

        params[i] = [H, aspect_ratio, k0, wall_thickness_mm, tau_h]

    return params


def check_public_relative_bounds(X_rel: np.ndarray, *, tol: float = 1e-10) -> None:
    """Check that all public relative parameters are inside [0, 1]."""
    X_rel = np.asarray(X_rel, dtype=float)
    if np.any(X_rel < -tol) or np.any(X_rel > 1.0 + tol):
        raise ValueError(
            "The parameter set is outside the investigated design space. "
            "All relative parameters must be within [0, 1]."
        )


def _check_single_parameter_choice(name: str, physical_value: Optional[float], relative_value: Optional[float]) -> None:
    if physical_value is None and relative_value is None:
        raise ValueError(f"Provide either {name} as a physical value or {name}_rel as a relative value.")

    if physical_value is not None and relative_value is not None:
        raise ValueError(f"Provide only one of {name} and {name}_rel, not both.")


def _resolve_mixed_input(
    *,
    H: Optional[float],
    H_rel: Optional[float],
    aspect_ratio: Optional[float],
    aspect_ratio_rel: Optional[float],
    k0: Optional[float],
    k0_rel: Optional[float],
    wall_thickness_mm: Optional[float],
    d_rel: Optional[float],
    tau_h: Optional[float],
    tau_rel: Optional[float],
) -> tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Resolve mixed physical/relative public inputs.

    For each parameter, exactly one representation must be given.
    The returned relative input uses the public relative convention:
        [H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel]
    """

    _check_single_parameter_choice("H", H, H_rel)
    _check_single_parameter_choice("aspect_ratio", aspect_ratio, aspect_ratio_rel)
    _check_single_parameter_choice("k0", k0, k0_rel)
    _check_single_parameter_choice("wall_thickness_mm", wall_thickness_mm, d_rel)
    _check_single_parameter_choice("tau_h", tau_h, tau_rel)

    if H is not None:
        H_abs = float(H)
        H_rel_value = (H_abs - 1.0) / (3.0 - 1.0)
    else:
        H_rel_value = float(H_rel)
        H_abs = 1.0 + H_rel_value * (3.0 - 1.0)

    if aspect_ratio is not None:
        aspect_ratio_abs = float(aspect_ratio)
        aspect_ratio_rel_value = (aspect_ratio_abs - 0.5) / (1.5 - 0.5)
    else:
        aspect_ratio_rel_value = float(aspect_ratio_rel)
        aspect_ratio_abs = 0.5 + aspect_ratio_rel_value * (1.5 - 0.5)

    if k0 is not None:
        k0_abs = float(k0)
        k0_rel_value = k0_to_k0_rel(k0_abs)
    else:
        k0_rel_value = float(k0_rel)
        k0_abs = k0_rel_to_k0(k0_rel_value)

    d_min_mm, d_max_mm, tau_min_h, tau_max_h = doe_bounds(H_abs, aspect_ratio_abs)

    if wall_thickness_mm is not None:
        wall_thickness_mm_abs = float(wall_thickness_mm)
        d_rel_value = (wall_thickness_mm_abs - d_min_mm) / (d_max_mm - d_min_mm)
    else:
        d_rel_value = float(d_rel)
        wall_thickness_mm_abs = d_min_mm + d_rel_value * (d_max_mm - d_min_mm)

    if tau_h is not None:
        tau_h_abs = float(tau_h)
        tau_rel_value = (tau_h_abs - tau_min_h) / (tau_max_h - tau_min_h)
    else:
        tau_rel_value = float(tau_rel)
        tau_h_abs = tau_min_h + tau_rel_value * (tau_max_h - tau_min_h)

    X_rel_public = np.array(
        [[H_rel_value, aspect_ratio_rel_value, k0_rel_value, d_rel_value, tau_rel_value]],
        dtype=float,
    )
    check_public_relative_bounds(X_rel_public)

    params_phys = np.array(
        [[H_abs, aspect_ratio_abs, k0_abs, wall_thickness_mm_abs, tau_h_abs]],
        dtype=float,
    )

    input_relative_public = {
        "H_rel": float(H_rel_value),
        "aspect_ratio_rel": float(aspect_ratio_rel_value),
        "k0_rel": float(k0_rel_value),
        "d_rel": float(d_rel_value),
        "tau_rel": float(tau_rel_value),
    }

    input_physical = {
        "H_m": float(H_abs),
        "aspect_ratio": float(aspect_ratio_abs),
        "k0": float(k0_abs),
        "wall_thickness_mm": float(wall_thickness_mm_abs),
        "tau_h": float(tau_h_abs),
    }

    return X_rel_public, params_phys, input_relative_public, input_physical


# -----------------------------------------------------------------------------
# Thermodynamic property polynomials and integration helpers
# -----------------------------------------------------------------------------


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for NumPy versions with/without trapezoid."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def rho_poly_np(T: np.ndarray | float) -> np.ndarray | float:
    """Temperature-dependent water density polynomial used by the model."""
    return (
        -6.76640181297590e-08 * T**4
        + 0.000100086952242332 * T**3
        - 0.0579435022428700 * T**2
        + 14.7754417118411 * T
        - 375.428322683367
    )


def cp_poly_np(T: np.ndarray | float) -> np.ndarray | float:
    """Temperature-dependent water heat-capacity polynomial used by the model."""
    return (
        3.49501206499032e-09 * T**6
        - 6.99935735700797e-06 * T**5
        + 0.005834531725005 * T**4
        - 2.59120258356208 * T**3
        + 646.658253168169 * T**2
        - 85983.4516356403 * T
        + 4763311.11316660
    )


def int_cp_np(T: np.ndarray | float) -> np.ndarray | float:
    """Analytical integral of cp(T)."""
    return (
        (3.49501206499032e-09 / 7) * T**7
        + (-6.99935735700797e-06 / 6) * T**6
        + (0.005834531725005 / 5) * T**5
        + (-2.59120258356208 / 4) * T**4
        + (646.658253168169 / 3) * T**3
        + (-85983.4516356403 / 2) * T**2
        + 4763311.11316660 * T
    )


def int_cp_over_T_np(T: np.ndarray | float) -> np.ndarray | float:
    """Analytical integral of cp(T)/T."""
    return (
        (3.49501206499032e-09 / 6) * T**6
        + (-6.99935735700797e-06 / 5) * T**5
        + (0.005834531725005 / 4) * T**4
        + (-2.59120258356208 / 3) * T**3
        + (646.658253168169 / 2) * T**2
        + (-85983.4516356403) * T
        + 4763311.11316660 * np.log(T)
    )


def effective_outer_heat_transfer_coefficient(
    *,
    H: float,
    aspect_ratio: float,
    tau_h: float,
    T_ref_K: float = 298.15,
) -> Dict[str, float]:
    """
    Calculate the effective outer heat-transfer coefficient from the storage time constant.

    The relation follows the lumped-capacitance expression used in the CFD setup:
        tau = rho * cp / alpha_eff * V/A

    Therefore:
        alpha_eff = rho * cp * (V/A) / tau

    with tau in seconds.
    """

    geom = storage_geometry(H, aspect_ratio)
    tau_s = float(tau_h) * 3600.0
    rho_ref = float(rho_poly_np(float(T_ref_K)))
    cp_ref = float(cp_poly_np(float(T_ref_K)))
    volume_to_surface_ratio_m = geom["volume_to_surface_ratio_m"]

    alpha_eff_W_m2K = rho_ref * cp_ref * volume_to_surface_ratio_m / tau_s

    return {
        "alpha_eff_W_m2K": float(alpha_eff_W_m2K),
        "T_ref_K": float(T_ref_K),
        "rho_ref_kg_m3": float(rho_ref),
        "cp_ref_J_kgK": float(cp_ref),
        "tau_h": float(tau_h),
        "tau_s": float(tau_s),
    }


def calculate_energy_exergy(
    T_time_z: np.ndarray,
    z_rel: np.ndarray,
    H: float,
    aspect_ratio: float,
    T0_K: float,
) -> Dict[str, np.ndarray]:
    """
    Calculate total absolute and normalized energy/exergy from temperature profiles.

    Parameters
    ----------
    T_time_z : ndarray, shape (n_time, n_z)
        Temperature profile over time and normalized height.
    z_rel : ndarray, shape (n_z,)
        Normalized vertical coordinate z/H.
    H : float
        Storage height in m.
    aspect_ratio : float
        Diameter-to-height ratio D/H.
    T0_K : float
        Dead-state/reference temperature in K.
    """

    T_time_z = np.asarray(T_time_z, dtype=np.float64)
    z_rel = np.asarray(z_rel, dtype=np.float64).ravel()

    geom = storage_geometry(H, aspect_ratio)
    area_cross = geom["cross_section_area_m2"]
    z_abs = z_rel * float(H)

    rho = rho_poly_np(T_time_z)
    dh = int_cp_np(T_time_z) - int_cp_np(T0_K)
    ds = int_cp_over_T_np(T_time_z) - int_cp_over_T_np(T0_K)
    ex_specific = dh - T0_K * ds

    mass_kg = area_cross * _trapz(rho, z_abs, axis=1)
    energy_J = area_cross * _trapz(rho * dh, z_abs, axis=1)
    exergy_J = area_cross * _trapz(rho * ex_specific, z_abs, axis=1)

    return {
        "mass_kg": mass_kg,
        "energy_J": energy_J,
        "energy_MJ": energy_J / 1.0e6,
        "energy_kWh": energy_J / 3.6e6,
        "exergy_J": exergy_J,
        "exergy_MJ": exergy_J / 1.0e6,
        "exergy_kWh": exergy_J / 3.6e6,
        "energy_norm": energy_J / energy_J[0],
        "exergy_norm": exergy_J / exergy_J[0],
    }


# -----------------------------------------------------------------------------
# Temperature profile reconstruction
# -----------------------------------------------------------------------------


def sigmoid_temperature(
    z_rel: np.ndarray,
    Tmax: np.ndarray,
    k: np.ndarray,
    x0: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Logistic vertical temperature profile convention used by the surrogate."""
    arg = np.clip(float(cfg.get("sigmoid_sign", -1.0)) * k * (z_rel - x0), -60.0, 60.0)
    T_amb_K = float(cfg.get("T_amb_K", 298.15))
    return T_amb_K + (Tmax - T_amb_K) / (1.0 + np.exp(arg))


def reconstruct_temperature_np(
    k_norm: np.ndarray,
    Tmax_norm: np.ndarray,
    x0_norm: np.ndarray,
    kfit0: float,
    Tmax0: float,
    x00: float,
    cfg: Dict[str, Any],
    *,
    z_rel: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct time-resolved vertical temperature profiles from MLP outputs."""

    if z_rel is None:
        z_rel = np.linspace(0.0, 1.0, int(cfg.get("n_z_eval", 300)))
    else:
        z_rel = np.asarray(z_rel, dtype=np.float64).ravel()

    k = np.asarray(k_norm, dtype=np.float64).reshape(-1, 1) * float(kfit0)
    Tmax = np.asarray(Tmax_norm, dtype=np.float64).reshape(-1, 1) * float(Tmax0)
    x0 = np.asarray(x0_norm, dtype=np.float64).reshape(-1, 1) * float(x00)

    T_profile = sigmoid_temperature(z_rel.reshape(1, -1), Tmax, k, x0, cfg)
    return z_rel, T_profile


# -----------------------------------------------------------------------------
# Surrogate loading and prediction
# -----------------------------------------------------------------------------


def load_surrogate(cache_path: str | Path, device: Optional[str | torch.device] = None) -> TESSurrogate:
    """
    Load the trained MLP surrogate from final_prediction_cache.pt.

    Parameters
    ----------
    cache_path : str or Path
        Path to the trained prediction cache.
    device : str or torch.device, optional
        "cpu", "cuda" or None. If None, CUDA is used when available.
    """

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Prediction cache not found: {cache_path}")

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    try:
        payload = torch.load(cache_path, map_location=device_obj, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=device_obj)

    cfg = payload["cfg"]
    out_dim = int(sum(cfg["n_pca"].values()))

    model = MLP(
        in_dim=5,
        out_dim=out_dim,
        hidden_dims=list(cfg["hidden_dims"]),
    ).to(device_obj)

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return TESSurrogate(
        model=model,
        pcas=payload["pcas"],
        x_scaler=payload["x_scaler"],
        y_scaler=payload["y_scaler"],
        t_s=np.asarray(payload["t"], dtype=np.float64).ravel(),
        cfg=cfg,
        device=device_obj,
    )


def _predict_normalized_curves(surrogate: TESSurrogate, X_mlp: np.ndarray) -> Dict[str, np.ndarray]:
    """Predict normalized sigmoid-parameter curves from internal MLP input."""

    X_mlp = np.atleast_2d(np.asarray(X_mlp, dtype=float))
    if np.any(X_mlp < -1e-10) or np.any(X_mlp > 1.0 + 1e-10):
        raise ValueError("Internal MLP input is outside [0, 1].")

    X_scaled = surrogate.x_scaler.transform(X_mlp)

    surrogate.model.eval()
    with torch.no_grad():
        Y_scaled = surrogate.model(
            torch.tensor(X_scaled, dtype=torch.float32, device=surrogate.device)
        ).cpu().numpy()

    Y = surrogate.y_scaler.inverse_transform(Y_scaled)

    pred: Dict[str, np.ndarray] = {}
    start = 0
    for name in surrogate.cfg["target_names"]:
        n_coeff = int(surrogate.cfg["n_pca"][name])
        end = start + n_coeff
        pred[name] = surrogate.pcas[name].inverse_transform(Y[:, start:end])
        start = end

    return pred


def _interp_at_time(time_s: np.ndarray, y: np.ndarray, target_time_s: float) -> np.ndarray | float:
    """Interpolate a scalar time series or time-profile array at target_time_s."""

    y = np.asarray(y, dtype=float)
    time_s = np.asarray(time_s, dtype=float).ravel()

    if y.ndim == 1:
        return float(np.interp(target_time_s, time_s, y))

    return np.array([np.interp(target_time_s, time_s, y[:, j]) for j in range(y.shape[1])])


def _build_prediction_result(
    surrogate: TESSurrogate,
    *,
    X_rel_public: np.ndarray,
    params_phys: np.ndarray,
    input_relative_public: Dict[str, float],
    input_physical: Dict[str, float],
    time_h: float,
    return_time_series: bool,
) -> Dict[str, Any]:
    """Shared prediction implementation once input values have been resolved."""

    target_time_s = float(time_h) * 3600.0
    if target_time_s < surrogate.t_s[0] or target_time_s > surrogate.t_s[-1]:
        raise ValueError(
            f"time_h={time_h} is outside the available prediction range "
            f"[{surrogate.t_s[0] / 3600.0:.3g}, {surrogate.t_s[-1] / 3600.0:.3g}] h."
        )

    H, aspect_ratio, k0, wall_thickness_mm, tau_h = params_phys.reshape(-1)

    X_mlp = public_relative_to_mlp_input(X_rel_public)
    pred = _predict_normalized_curves(surrogate, X_mlp)

    z_rel, T_time_z = reconstruct_temperature_np(
        pred["k_norm"][0],
        pred["Tmax_norm"][0],
        pred["x0_norm"][0],
        kfit0=float(k0),
        Tmax0=float(surrogate.cfg.get("default_Tmax0_K", 368.15)),
        x00=float(surrogate.cfg.get("default_x00", 0.5)),
        cfg=surrogate.cfg,
    )

    T0_K = float(surrogate.cfg.get("T0_K", 298.15))
    thermo = calculate_energy_exergy(
        T_time_z=T_time_z,
        z_rel=z_rel,
        H=float(H),
        aspect_ratio=float(aspect_ratio),
        T0_K=T0_K,
    )

    geom = storage_geometry(float(H), float(aspect_ratio))
    heat_transfer = effective_outer_heat_transfer_coefficient(
        H=float(H),
        aspect_ratio=float(aspect_ratio),
        tau_h=float(tau_h),
        T_ref_K=T0_K,
    )

    z_abs_m = z_rel * float(H)
    T_at_time = _interp_at_time(surrogate.t_s, T_time_z, target_time_s)

    # Add the derived scalar quantities also to input_physical for convenience.
    input_physical_extended = dict(input_physical)
    input_physical_extended.update(
        {
            "diameter_m": geom["diameter_m"],
            "radius_m": geom["radius_m"],
            "cross_section_area_m2": geom["cross_section_area_m2"],
            "outer_surface_area_m2": geom["outer_surface_area_m2"],
            "volume_m3": geom["volume_m3"],
            "volume_to_surface_ratio_m": geom["volume_to_surface_ratio_m"],
            "alpha_eff_W_m2K": heat_transfer["alpha_eff_W_m2K"],
        }
    )

    result: Dict[str, Any] = {
        "time_h": float(time_h),
        "input_relative": input_relative_public,
        "input_physical": input_physical_extended,
        "geometry": geom,
        "heat_transfer": heat_transfer,
        "z_rel": z_rel,
        "z_abs_m": z_abs_m,
        "temperature_K": T_at_time,
        "energy": {
            "J": _interp_at_time(surrogate.t_s, thermo["energy_J"], target_time_s),
            "MJ": _interp_at_time(surrogate.t_s, thermo["energy_MJ"], target_time_s),
            "kWh": _interp_at_time(surrogate.t_s, thermo["energy_kWh"], target_time_s),
            "normalized": _interp_at_time(surrogate.t_s, thermo["energy_norm"], target_time_s),
        },
        "exergy": {
            "J": _interp_at_time(surrogate.t_s, thermo["exergy_J"], target_time_s),
            "MJ": _interp_at_time(surrogate.t_s, thermo["exergy_MJ"], target_time_s),
            "kWh": _interp_at_time(surrogate.t_s, thermo["exergy_kWh"], target_time_s),
            "normalized": _interp_at_time(surrogate.t_s, thermo["exergy_norm"], target_time_s),
        },
        "mass_kg": _interp_at_time(surrogate.t_s, thermo["mass_kg"], target_time_s),
    }

    if return_time_series:
        result["time_series"] = {
            "time_s": surrogate.t_s,
            "time_h": surrogate.t_s / 3600.0,
            "temperature_K": T_time_z,
            "energy_J": thermo["energy_J"],
            "energy_MJ": thermo["energy_MJ"],
            "energy_kWh": thermo["energy_kWh"],
            "energy_norm": thermo["energy_norm"],
            "exergy_J": thermo["exergy_J"],
            "exergy_MJ": thermo["exergy_MJ"],
            "exergy_kWh": thermo["exergy_kWh"],
            "exergy_norm": thermo["exergy_norm"],
            "mass_kg": thermo["mass_kg"],
        }

    return result


def predict_tes_from_relative(
    surrogate: TESSurrogate,
    X_rel: np.ndarray,
    *,
    time_h: float,
    return_time_series: bool = False,
) -> Dict[str, Any]:
    """
    Predict TES temperature profile and thermodynamic performance from public relative input.

    Public relative input order:
        [H_rel, aspect_ratio_rel, k0_rel, d_rel, tau_rel]

    The logarithmic transformation of k0 is performed internally before the MLP
    is evaluated.
    """

    X_rel_public = np.asarray(X_rel, dtype=float).reshape(1, 5)
    check_public_relative_bounds(X_rel_public)

    params_phys = relative_to_physical(X_rel_public)
    H, aspect_ratio, k0, wall_thickness_mm, tau_h = params_phys[0]

    input_relative_public = {
        "H_rel": float(X_rel_public[0, 0]),
        "aspect_ratio_rel": float(X_rel_public[0, 1]),
        "k0_rel": float(X_rel_public[0, 2]),
        "d_rel": float(X_rel_public[0, 3]),
        "tau_rel": float(X_rel_public[0, 4]),
    }

    input_physical = {
        "H_m": float(H),
        "aspect_ratio": float(aspect_ratio),
        "k0": float(k0),
        "wall_thickness_mm": float(wall_thickness_mm),
        "tau_h": float(tau_h),
    }

    return _build_prediction_result(
        surrogate,
        X_rel_public=X_rel_public,
        params_phys=params_phys,
        input_relative_public=input_relative_public,
        input_physical=input_physical,
        time_h=time_h,
        return_time_series=return_time_series,
    )


def predict_tes(
    surrogate: TESSurrogate,
    *,
    time_h: float,
    H: Optional[float] = None,
    H_rel: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    aspect_ratio_rel: Optional[float] = None,
    k0: Optional[float] = None,
    k0_rel: Optional[float] = None,
    wall_thickness_mm: Optional[float] = None,
    d_rel: Optional[float] = None,
    tau_h: Optional[float] = None,
    tau_rel: Optional[float] = None,
    return_time_series: bool = False,
) -> Dict[str, Any]:
    """
    Predict TES temperature profile and thermodynamic performance.

    For each input parameter, either a physical value or its relative value can be
    provided. Do not provide both representations for the same parameter.

    Examples
    --------
    Fully physical input:
        predict_tes(
            model,
            H=2.0,
            aspect_ratio=1.0,
            k0=55.0,
            wall_thickness_mm=5.5,
            tau_h=1200.0,
            time_h=24.0,
        )

    Fully relative input:
        predict_tes(
            model,
            H_rel=0.5,
            aspect_ratio_rel=0.5,
            k0_rel=0.5,
            d_rel=0.5,
            tau_rel=0.5,
            time_h=24.0,
        )

    Mixed input:
        predict_tes(
            model,
            H=2.0,
            aspect_ratio=1.0,
            k0_rel=0.5,
            d_rel=0.2,
            tau_rel=0.8,
            time_h=24.0,
        )

    Returned dictionary contains:
        result["temperature_K"]
        result["energy"]
        result["exergy"]
        result["geometry"]
        result["heat_transfer"]
    """

    X_rel_public, params_phys, input_relative_public, input_physical = _resolve_mixed_input(
        H=H,
        H_rel=H_rel,
        aspect_ratio=aspect_ratio,
        aspect_ratio_rel=aspect_ratio_rel,
        k0=k0,
        k0_rel=k0_rel,
        wall_thickness_mm=wall_thickness_mm,
        d_rel=d_rel,
        tau_h=tau_h,
        tau_rel=tau_rel,
    )

    return _build_prediction_result(
        surrogate,
        X_rel_public=X_rel_public,
        params_phys=params_phys,
        input_relative_public=input_relative_public,
        input_physical=input_physical,
        time_h=time_h,
        return_time_series=return_time_series,
    )


#%%

# -----------------------------------------------------------------------------
# Minimal command-line example
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    model = load_surrogate("models/final_prediction_cache.pt", device="cpu")

    result = predict_tes(
        model,
        H=2,
        aspect_ratio=0.58,
        k0=30.59,
        wall_thickness_mm=5.0,
        tau_h=886,
        time_h=24.0,
    )

    print("Predicted values at t = 24 h")
    print(f"Energy: {result['energy']['kWh']:.6g} kWh")
    print(f"Exergy: {result['exergy']['kWh']:.6g} kWh")
    print(f"Normalized energy: {result['energy']['normalized']:.6g}")
    print(f"Normalized exergy: {result['exergy']['normalized']:.6g}")
    print(f"Storage volume: {result['geometry']['volume_m3']:.6g} m^3")
    print(f"Storage diameter: {result['geometry']['diameter_m']:.6g} m")
    print(f"V/A: {result['geometry']['volume_to_surface_ratio_m']:.6g} m")
    print(f"alpha_eff: {result['heat_transfer']['alpha_eff_W_m2K']:.6g} W/(m^2 K)")
    