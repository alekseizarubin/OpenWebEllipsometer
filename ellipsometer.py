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
    air_fraction: float = 0.0

    def complex_n(self) -> complex:
        n_comp = complex(self.n, self.k)
        if self.air_fraction > 0:
            n_air = complex(1.0, 0.0)
            n_comp = self.air_fraction * n_air + (1 - self.air_fraction) * n_comp
        return n_comp

@dataclass
class ModelParams:
    n_before: float = 1.0
    layers: List[Layer] = field(default_factory=list)
    n_sub: float = 1.5
    k_sub: float = 0.0
    intensity_scale: float = 1.0
    intensity_offset: float = 0.0
    polarizer_deg: float = 45.0
    incidence_offset_deg: float = 0.0
    analyzer_offset_deg: float = 0.0

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
    theta0 = np.deg2rad(inc_deg + params.incidence_offset_deg)
    n_list = [complex(params.n_before, 0.0)]
    d_list = []
    for layer in params.layers:
        n_list.append(layer.complex_n())
        d_list.append(layer.thickness_nm * 1e-9)
    n_list.append(params.substrate_n())

    r_s = _layer_r(wl, theta0, n_list, d_list, "s")
    r_p = _layer_r(wl, theta0, n_list, d_list, "p")

    pol = np.deg2rad(params.polarizer_deg)
    ana = np.deg2rad(analyzer_deg + params.analyzer_offset_deg)

    inc_vec = np.array([np.cos(pol), np.sin(pol)])
    ref_vec = np.array([r_p * inc_vec[0], r_s * inc_vec[1]])

    cross = np.array([-np.sin(pol), np.cos(pol)])
    rot = np.array([[np.cos(ana), -np.sin(ana)], [np.sin(ana), np.cos(ana)]])
    ana_vec = rot @ cross

    E = ref_vec[0] * ana_vec[0] + ref_vec[1] * ana_vec[1]
    return params.intensity_offset + params.intensity_scale * (np.abs(E) ** 2)


def group_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Group measurement rows and compute averages and extremes."""
    # ensure intensity values are numeric to avoid aggregation errors
    df = df.copy()
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna(subset=["intensity"])

    grouped = df.groupby(["wavelength_nm", "incidence_deg", "analyzer_deg"])
    out = grouped["intensity"].agg(["mean", "min", "max"]).reset_index()
    out = out.rename(columns={"mean": "intensity_mean", "min": "intensity_min", "max": "intensity_max"})
    return out


def normalise_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise intensities per ``(wavelength_nm, incidence_deg)`` pair.

    Each group is shifted so its minimum becomes zero and scaled by the
    range within that group. If the range is zero the values remain
    unchanged. This preserves relative shapes while allowing a global
    intensity scale factor to be fitted later.
    """

    df = df.copy()
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna(subset=["intensity"])

    grp = df.groupby(["wavelength_nm", "incidence_deg"])
    baseline = grp["intensity"].transform("min")
    max_val = grp["intensity"].transform("max")
    scale = max_val - baseline
    scale[scale <= 0] = 1.0

    df["intensity"] = (df["intensity"] - baseline) / scale
    return df


def predict_dataframe(df: pd.DataFrame, params: ModelParams) -> np.ndarray:
    pred = [
        analyzer_intensity(row.wavelength_nm, row.incidence_deg, row.analyzer_deg, params)
        for row in df.itertuples(index=False)
    ]
    return np.asarray(pred)


def fit_parameters(
    df: pd.DataFrame,
    params: ModelParams,
    optimise: Dict[str, bool],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[ModelParams, float]:
    """Fit unknown parameters using least squares.

    Parameters
    ----------
    df : DataFrame
        Grouped measurement data produced by :func:`group_measurements`.
    params : ModelParams
        Initial parameters. Parameters not marked for optimisation are fixed.
    optimise : dict
        Keys should have the form 'layer{i}_n', 'layer{i}_k', 'layer{i}_d'
        for each layer index ``i`` starting from 0, 'sub_n', 'sub_k' for the
        substrate and 'scale' for the intensity scale factor. Additional keys
        include 'offset' for a constant intensity offset, 'polarizer_deg' for
        the polarisation plane, 'inc_offset' for an incidence angle offset and
        'an_offset' for an analyzer angle offset.
    bounds : dict, optional
        Mapping from the same keys as ``optimise`` to ``(min, max)`` tuples.
        If omitted, reasonable defaults are used.
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

    def _get_bounds(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        if bounds and key in bounds:
            return bounds[key]
        return default

    for i, layer in enumerate(params.layers):
        if optimise.get(f"layer{i}_n", False):
            x0.append(layer.n)
            lo, hi = _get_bounds(f"layer{i}_n", (0.0, np.inf))
            bounds_lo.append(lo)
            bounds_hi.append(hi)
            path.append(("layer", i, "n"))
        if optimise.get(f"layer{i}_k", False):
            x0.append(layer.k)
            lo, hi = _get_bounds(f"layer{i}_k", (0.0, np.inf))
            bounds_lo.append(lo)
            bounds_hi.append(hi)
            path.append(("layer", i, "k"))
        if optimise.get(f"layer{i}_d", False):
            x0.append(layer.thickness_nm)
            lo, hi = _get_bounds(f"layer{i}_d", (0.0, 1000.0))
            bounds_lo.append(lo)
            bounds_hi.append(hi)
            path.append(("layer", i, "d"))
    if optimise.get("sub_n", False):
        x0.append(params.n_sub)
        lo, hi = _get_bounds("sub_n", (0.0, np.inf))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("sub", "n"))
    if optimise.get("sub_k", False):
        x0.append(params.k_sub)
        lo, hi = _get_bounds("sub_k", (0.0, np.inf))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("sub", "k"))
    if optimise.get("scale", False):
        x0.append(params.intensity_scale)
        lo, hi = _get_bounds("scale", (0.0, np.inf))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("scale",))
    if optimise.get("offset", False):
        x0.append(params.intensity_offset)
        lo, hi = _get_bounds("offset", (-np.inf, np.inf))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("offset",))
    if optimise.get("polarizer_deg", False):
        x0.append(params.polarizer_deg)
        lo, hi = _get_bounds("polarizer_deg", (0.0, 180.0))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("polarizer_deg",))
    if optimise.get("inc_offset", False):
        x0.append(params.incidence_offset_deg)
        lo, hi = _get_bounds("inc_offset", (-10.0, 10.0))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("inc_offset",))
    if optimise.get("an_offset", False):
        x0.append(params.analyzer_offset_deg)
        lo, hi = _get_bounds("an_offset", (-10.0, 10.0))
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        path.append(("an_offset",))

    def unpack(x, p: ModelParams) -> ModelParams:
        p = ModelParams(
            n_before=p.n_before,
            layers=[Layer(l.n, l.k, l.thickness_nm, l.air_fraction) for l in p.layers],
            n_sub=p.n_sub,
            k_sub=p.k_sub,
            intensity_scale=p.intensity_scale,
            intensity_offset=p.intensity_offset,
            polarizer_deg=p.polarizer_deg,
            incidence_offset_deg=p.incidence_offset_deg,
            analyzer_offset_deg=p.analyzer_offset_deg,
        )
        for val, entry in zip(x, path):
            if entry[0] == "layer":
                idx = entry[1]
                field = entry[2]
                if field == "n":
                    p.layers[idx].n = val
                elif field == "k":
                    p.layers[idx].k = val
                else:
                    p.layers[idx].thickness_nm = val
            elif entry[0] == "sub":
                if entry[1] == "n":
                    p.n_sub = val
                else:
                    p.k_sub = val
            elif entry[0] == "scale":
                p.intensity_scale = val
            elif entry[0] == "offset":
                p.intensity_offset = val
            elif entry[0] == "polarizer_deg":
                p.polarizer_deg = val
            elif entry[0] == "inc_offset":
                p.incidence_offset_deg = val
            elif entry[0] == "an_offset":
                p.analyzer_offset_deg = val
        return p

    def residual(x):
        p = unpack(x, params)
        pred = predict_dataframe(df, p)
        return pred - df.intensity_mean.to_numpy()

    result = least_squares(residual, x0, bounds=(bounds_lo, bounds_hi))
    final_params = unpack(result.x, params)
    rmse = np.sqrt(np.mean(residual(result.x) ** 2))
    return final_params, rmse


def calibrate_system(
    df: pd.DataFrame,
    params: ModelParams,
    optimise: Optional[Dict[str, bool]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[ModelParams, float]:
    """Determine system angle offsets using a reference sample."""
    if optimise is None:
        optimise = {
            "polarizer_deg": True,
            "inc_offset": True,
            "an_offset": True,
            "offset": True,
            "scale": True,
        }
    grouped = group_measurements(df)
    return fit_parameters(grouped, params, optimise, bounds)
