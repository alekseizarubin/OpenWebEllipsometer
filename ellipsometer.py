import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import least_squares

@dataclass
class Layer:
    n: float
    k: float = 0.0
    thickness_nm: float = 0.0

    def complex_n(self) -> complex:
        return complex(self.n, self.k)

@dataclass
class ModelParams:
    n_before: float = 1.0
    layers: List[Layer] = field(default_factory=list)
    n_sub: float = 1.5
    k_sub: float = 0.0

    def substrate_n(self) -> complex:
        return complex(self.n_sub, self.k_sub)

# ---------------------------------------------------------------------------
# Utility functions for thin film calculations
# ---------------------------------------------------------------------------

def _cos_theta(n0: complex, n: complex, theta0: float) -> complex:
    """Cosine of angle inside material with index n."""
    sin_t = n0 * np.sin(theta0) / n
    return np.sqrt(1 - sin_t ** 2)


def _r_interface(n_i: complex, n_j: complex, c_i: complex, c_j: complex, pol: str) -> complex:
    if pol == "s":
        num = n_i * c_i - n_j * c_j
        den = n_i * c_i + n_j * c_j
    else:  # p polarization
        num = n_j * c_i - n_i * c_j
        den = n_j * c_i + n_i * c_j
    return num / den


def _layer_r(wl: float, theta0: float, n_list: List[complex], d_list: List[float], pol: str) -> complex:
    c_list = [_cos_theta(n_list[0], n, theta0) for n in n_list]
    r = _r_interface(n_list[-2], n_list[-1], c_list[-2], c_list[-1], pol)
    for i in range(len(d_list) - 1, -1, -1):
        delta = 2 * np.pi * n_list[i + 1] * d_list[i] * c_list[i + 1] / wl
        r_int = _r_interface(n_list[i], n_list[i + 1], c_list[i], c_list[i + 1], pol)
        exp_term = np.exp(2j * delta)
        r = (r_int + r * exp_term) / (1 + r_int * r * exp_term)
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyzer_intensity(wl_nm: float, inc_deg: float, analyzer_deg: float, params: ModelParams) -> float:
    """Calculate reflected intensity after analyzer.

    Parameters
    ----------
    wl_nm : float
        Wavelength in nanometers.
    inc_deg : float
        Angle of incidence in degrees (0 = normal).
    analyzer_deg : float
        Analyzer rotation angle in degrees.
    params : ModelParams
        Optical stack parameters.
    """
    wl = wl_nm * 1e-9
    theta0 = np.deg2rad(inc_deg)
    n_list = [complex(params.n_before, 0.0)]
    d_list = []
    for layer in params.layers:
        n_list.append(layer.complex_n())
        d_list.append(layer.thickness_nm * 1e-9)
    n_list.append(params.substrate_n())

    r_s = _layer_r(wl, theta0, n_list, d_list, "s")
    r_p = _layer_r(wl, theta0, n_list, d_list, "p")

    A = np.deg2rad(analyzer_deg)
    E = r_s * np.cos(A) + r_p * np.sin(A)
    return np.abs(E) ** 2


def group_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Group measurement rows and compute averages and extremes."""
    grouped = df.groupby(["wavelength_nm", "incidence_deg", "analyzer_deg"])
    out = grouped["intensity"].agg(["mean", "min", "max"]).reset_index()
    out = out.rename(columns={"mean": "intensity_mean", "min": "intensity_min", "max": "intensity_max"})
    return out


def predict_dataframe(df: pd.DataFrame, params: ModelParams) -> np.ndarray:
    pred = [
        analyzer_intensity(row.wavelength_nm, row.incidence_deg, row.analyzer_deg, params)
        for row in df.itertuples(index=False)
    ]
    return np.asarray(pred)


def fit_parameters(df: pd.DataFrame, params: ModelParams, optimise: Dict[str, bool]) -> Tuple[ModelParams, float]:
    """Fit unknown parameters using least squares.

    Parameters
    ----------
    df : DataFrame
        Grouped measurement data produced by :func:`group_measurements`.
    params : ModelParams
        Initial parameters. Parameters not marked for optimisation are fixed.
    optimise : dict
        Dictionary with keys 'layer_n', 'layer_k', 'layer_d', 'sub_n', 'sub_k'.
    Returns
    -------
    ModelParams
        Optimised parameters.
    float
        Root mean squared error of the fit.
    """
    x0 = []
    bounds_lo = []
    bounds_hi = []
    path = []

    if optimise.get("layer_n", False):
        x0.append(params.layers[0].n)
        bounds_lo.append(0)
        bounds_hi.append(5)
        path.append(("layer", "n"))
    if optimise.get("layer_k", False):
        x0.append(params.layers[0].k)
        bounds_lo.append(0)
        bounds_hi.append(5)
        path.append(("layer", "k"))
    if optimise.get("layer_d", False):
        x0.append(params.layers[0].thickness_nm)
        bounds_lo.append(0)
        bounds_hi.append(1000)
        path.append(("layer", "d"))
    if optimise.get("sub_n", False):
        x0.append(params.n_sub)
        bounds_lo.append(0)
        bounds_hi.append(5)
        path.append(("sub", "n"))
    if optimise.get("sub_k", False):
        x0.append(params.k_sub)
        bounds_lo.append(0)
        bounds_hi.append(5)
        path.append(("sub", "k"))

    def unpack(x, p: ModelParams) -> ModelParams:
        p = ModelParams(
            n_before=p.n_before,
            layers=[Layer(layer.n, layer.k, layer.thickness_nm) for layer in p.layers],
            n_sub=p.n_sub,
            k_sub=p.k_sub,
        )
        for val, entry in zip(x, path):
            if entry[0] == "layer":
                if entry[1] == "n":
                    p.layers[0].n = val
                elif entry[1] == "k":
                    p.layers[0].k = val
                else:
                    p.layers[0].thickness_nm = val
            elif entry[0] == "sub":
                if entry[1] == "n":
                    p.n_sub = val
                else:
                    p.k_sub = val
        return p

    def residual(x):
        p = unpack(x, params)
        pred = predict_dataframe(df, p)
        return pred - df.intensity_mean.to_numpy()

    result = least_squares(residual, x0, bounds=(bounds_lo, bounds_hi))
    final_params = unpack(result.x, params)
    rmse = np.sqrt(np.mean(residual(result.x) ** 2))
    return final_params, rmse
