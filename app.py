import streamlit as st
import pandas as pd
import numpy as np
from ellipsometer import (
    Layer,
    ModelParams,
    group_measurements,
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

uploaded = st.file_uploader("Upload measurement TSV", type="tsv", key="uploader")
if uploaded is not None:
    df_up = pd.read_csv(uploaded, sep="\t", decimal=".")
    st.session_state["meas_df"] = pd.concat(
        [st.session_state["meas_df"], df_up], ignore_index=True
    )
    st.session_state["uploader"] = None
    _rerun()

df = st.session_state["meas_df"].copy()
if df.empty:
    st.stop()

grouped = group_measurements(df)
st.subheader("Summary data")
st.dataframe(grouped)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Model parameters")

n_before = st.sidebar.number_input("n before film", value=1.0)

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
        opt_k = st.checkbox("fit k", value=False, key=f"opt_k_{i}")
        opt_d = st.checkbox("fit thickness", value=False, key=f"opt_d_{i}")
        st.session_state["layers"][i] = Layer(n_val, k_val, d_val, af_val)
        optimise[f"layer{i}_n"] = opt_n
        optimise[f"layer{i}_k"] = opt_k
        optimise[f"layer{i}_d"] = opt_d

with st.sidebar.expander("Substrate"):
    n_sub = st.number_input("substrate n", value=1.45)
    k_sub = st.number_input("substrate k", value=0.0)
    opt_n_sub = st.checkbox("fit substrate n", value=False)
    opt_k_sub = st.checkbox("fit substrate k", value=False)

params = ModelParams(
    n_before=n_before,
    layers=st.session_state["layers"],
    n_sub=n_sub,
    k_sub=k_sub,
)

optimise["sub_n"] = opt_n_sub
optimise["sub_k"] = opt_k_sub

if st.button("Start optimisation"):
    fitted, rmse = fit_parameters(grouped, params, optimise)
    st.subheader("Fit results")
    for i, layer in enumerate(fitted.layers):
        st.write(f"layer {i + 1} n: {layer.n:.4f}")
        st.write(f"layer {i + 1} k: {layer.k:.4f}")
        st.write(f"layer {i + 1} thickness [nm]: {layer.thickness_nm:.2f}")
    st.write(f"substrate n: {fitted.n_sub:.4f}")
    st.write(f"substrate k: {fitted.k_sub:.4f}")
    st.write(f"RMSE: {rmse:.6f}")

    preds = predict_dataframe(grouped, fitted)
    df_out = grouped.copy()
    df_out["model_intensity"] = preds
    st.subheader("Measurements vs. model")
    st.dataframe(df_out)
