import streamlit as st
import pandas as pd
import numpy as np
from ellipsometer import (
    Layer,
    ModelParams,
    group_measurements,
    normalise_measurements,
    predict_dataframe,
    fit_parameters,
)


def _rerun():
    """Compatibility wrapper for Streamlit rerun."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    elif hasattr(st, "rerun"):
        st.rerun()
    else:
        raise RuntimeError("Streamlit rerun API not available")

st.title("DIY-Ellipsometry fitter")

# ---------------------------------------------------------------------------
# Data input
# ---------------------------------------------------------------------------

if "meas_df" not in st.session_state:
    st.session_state["meas_df"] = pd.DataFrame(
        columns=["wavelength_nm", "incidence_deg", "analyzer_deg", "intensity"]
    )

data_edit = st.data_editor(
    st.session_state["meas_df"],
    num_rows="dynamic",
    key="meas_table",
    use_container_width=True,
)
st.session_state["meas_df"] = data_edit

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded = st.file_uploader(
    "Upload measurement TSV", type="tsv", key=f"uploader_{st.session_state.uploader_key}"
)
if uploaded is not None:
    df_up = pd.read_csv(uploaded, sep="\t", decimal=".")
    st.session_state["meas_df"] = pd.concat(
        [st.session_state["meas_df"], df_up], ignore_index=True
    )
    st.session_state["uploader_key"] += 1
    _rerun()

df = st.session_state["meas_df"].copy()
if df.empty:
    st.stop()

norm = st.sidebar.checkbox("normalise intensities", value=False)
if norm:
    df = normalise_measurements(df)

grouped = group_measurements(df)
st.subheader("Summary data")
st.dataframe(grouped)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Model parameters")

n_before = st.sidebar.number_input("n before film", value=1.0)
scale_val = st.sidebar.number_input("intensity scale", value=1.0)
opt_scale = st.sidebar.checkbox("fit intensity scale", value=False)
offset_val = st.sidebar.number_input("intensity offset", value=0.0)
opt_offset = st.sidebar.checkbox("fit intensity offset", value=False)
pol_angle = st.sidebar.number_input("polariser angle deg", value=45.0)
opt_pol = st.sidebar.checkbox("fit polariser angle", value=False)
inc_off = st.sidebar.number_input("incidence offset deg", value=0.0)
opt_inc_off = st.sidebar.checkbox("fit incidence offset", value=False)
an_off = st.sidebar.number_input("analyser offset deg", value=0.0)
opt_an_off = st.sidebar.checkbox("fit analyser offset", value=False)

if "layers" not in st.session_state:
    st.session_state["layers"] = []

col_add, col_remove = st.sidebar.columns(2)
if col_add.button("Add layer"):
    st.session_state["layers"].append(Layer(1.5, 0.0, 50.0))
    _rerun()
if col_remove.button("Remove layer") and st.session_state["layers"]:
    st.session_state["layers"].pop()
    _rerun()

optimise = {}
bounds = {}
for i, layer in enumerate(st.session_state["layers"]):
    with st.sidebar.expander(f"Layer {i + 1}"):
        n_val = st.number_input("n", value=layer.n, key=f"n_{i}")
        k_val = st.number_input("k", value=layer.k, key=f"k_{i}")
        d_val = st.number_input("thickness nm", value=layer.thickness_nm, key=f"d_{i}")
        af_val = st.number_input(
            "air fraction",
            value=layer.air_fraction,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key=f"af_{i}",
        )
        opt_n = st.checkbox("fit n", value=False, key=f"opt_n_{i}")
        if opt_n:
            n_min = st.number_input("n min", value=0.0, key=f"n_min_{i}")
            n_max = st.number_input("n max", value=10.0, key=f"n_max_{i}")
            bounds[f"layer{i}_n"] = (n_min, n_max)
        opt_k = st.checkbox("fit k", value=False, key=f"opt_k_{i}")
        if opt_k:
            k_min = st.number_input("k min", value=0.0, key=f"k_min_{i}")
            k_max = st.number_input("k max", value=10.0, key=f"k_max_{i}")
            bounds[f"layer{i}_k"] = (k_min, k_max)
        opt_d = st.checkbox("fit thickness", value=False, key=f"opt_d_{i}")
        if opt_d:
            d_min = st.number_input("thickness min", value=0.0, key=f"d_min_{i}")
            d_max = st.number_input("thickness max", value=1000.0, key=f"d_max_{i}")
            bounds[f"layer{i}_d"] = (d_min, d_max)
        st.session_state["layers"][i] = Layer(n_val, k_val, d_val, af_val)
        optimise[f"layer{i}_n"] = opt_n
        optimise[f"layer{i}_k"] = opt_k
        optimise[f"layer{i}_d"] = opt_d

with st.sidebar.expander("Substrate"):
    n_sub = st.number_input("substrate n", value=1.45)
    k_sub = st.number_input("substrate k", value=0.0)
    opt_n_sub = st.checkbox("fit substrate n", value=False)
    if opt_n_sub:
        sub_n_min = st.number_input("sub n min", value=0.0)
        sub_n_max = st.number_input("sub n max", value=10.0)
        bounds["sub_n"] = (sub_n_min, sub_n_max)
    opt_k_sub = st.checkbox("fit substrate k", value=False)
    if opt_k_sub:
        sub_k_min = st.number_input("sub k min", value=0.0)
        sub_k_max = st.number_input("sub k max", value=10.0)
        bounds["sub_k"] = (sub_k_min, sub_k_max)

params = ModelParams(
    n_before=n_before,
    layers=st.session_state["layers"],
    n_sub=n_sub,
    k_sub=k_sub,
    intensity_scale=scale_val,
    intensity_offset=offset_val,
    polarizer_deg=pol_angle,
    incidence_offset_deg=inc_off,
    analyzer_offset_deg=an_off,
)

optimise["sub_n"] = opt_n_sub
optimise["sub_k"] = opt_k_sub
optimise["scale"] = opt_scale
optimise["offset"] = opt_offset
optimise["polarizer_deg"] = opt_pol
optimise["inc_offset"] = opt_inc_off
optimise["an_offset"] = opt_an_off
if opt_scale:
    scale_min = st.sidebar.number_input("scale min", value=0.0)
    scale_max = st.sidebar.number_input("scale max", value=100000.0)
    bounds["scale"] = (scale_min, scale_max)
if opt_offset:
    off_min = st.sidebar.number_input("offset min", value=-1000.0)
    off_max = st.sidebar.number_input("offset max", value=1000.0)
    bounds["offset"] = (off_min, off_max)
if opt_pol:
    pol_min = st.sidebar.number_input("pol angle min", value=0.0)
    pol_max = st.sidebar.number_input("pol angle max", value=90.0)
    bounds["polarizer_deg"] = (pol_min, pol_max)
if opt_inc_off:
    inc_min = st.sidebar.number_input("inc offset min", value=-10.0)
    inc_max = st.sidebar.number_input("inc offset max", value=10.0)
    bounds["inc_offset"] = (inc_min, inc_max)
if opt_an_off:
    an_min = st.sidebar.number_input("an offset min", value=-10.0)
    an_max = st.sidebar.number_input("an offset max", value=10.0)
    bounds["an_offset"] = (an_min, an_max)

if st.button("Start optimisation"):
    fitted, rmse = fit_parameters(grouped, params, optimise, bounds)
    st.subheader("Fit results")
    for i, layer in enumerate(fitted.layers):
        st.write(f"layer {i + 1} n: {layer.n:.4f}")
        st.write(f"layer {i + 1} k: {layer.k:.4f}")
        st.write(f"layer {i + 1} thickness [nm]: {layer.thickness_nm:.2f}")
    st.write(f"substrate n: {fitted.n_sub:.4f}")
    st.write(f"substrate k: {fitted.k_sub:.4f}")
    st.write(f"intensity scale: {fitted.intensity_scale:.4f}")
    st.write(f"intensity offset: {fitted.intensity_offset:.4f}")
    st.write(f"polariser angle [deg]: {fitted.polarizer_deg:.2f}")
    st.write(f"incidence offset [deg]: {fitted.incidence_offset_deg:.2f}")
    st.write(f"analyser offset [deg]: {fitted.analyzer_offset_deg:.2f}")
    st.write(f"RMSE: {rmse:.6f}")

    preds = predict_dataframe(grouped, fitted)
    df_out = grouped.copy()
    df_out["model_intensity"] = preds
    st.subheader("Measurements vs. model")
    st.dataframe(df_out)
