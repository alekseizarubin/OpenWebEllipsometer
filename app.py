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

uploaded = st.file_uploader("Upload measurement CSV", type="csv")
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.session_state["meas_df"] = pd.concat([st.session_state["meas_df"], df_up])
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

with st.sidebar.expander("Thin film"):
    n_layer = st.number_input("film n", value=1.7)
    k_layer = st.number_input("film k", value=0.0)
    d_layer = st.number_input("thickness, nm", value=50.0)
    opt_n_layer = st.checkbox("fit n", value=False)
    opt_k_layer = st.checkbox("fit k", value=False)
    opt_d_layer = st.checkbox("fit thickness", value=False)

with st.sidebar.expander("Substrate"):
    n_sub = st.number_input("substrate n", value=1.45)
    k_sub = st.number_input("substrate k", value=0.0)
    opt_n_sub = st.checkbox("fit substrate n", value=False)
    opt_k_sub = st.checkbox("fit substrate k", value=False)

params = ModelParams(
    n_before=n_before,
    layers=[Layer(n_layer, k_layer, d_layer)],
    n_sub=n_sub,
    k_sub=k_sub,
)

optimise = {
    "layer_n": opt_n_layer,
    "layer_k": opt_k_layer,
    "layer_d": opt_d_layer,
    "sub_n": opt_n_sub,
    "sub_k": opt_k_sub,
}

if st.button("Start optimisation"):
    fitted, rmse = fit_parameters(grouped, params, optimise)
    st.subheader("Fit results")
    st.write(f"film n: {fitted.layers[0].n:.4f}")
    st.write(f"film k: {fitted.layers[0].k:.4f}")
    st.write(f"film thickness [nm]: {fitted.layers[0].thickness_nm:.2f}")
    st.write(f"substrate n: {fitted.n_sub:.4f}")
    st.write(f"substrate k: {fitted.k_sub:.4f}")
    st.write(f"RMSE: {rmse:.6f}")

    preds = predict_dataframe(grouped, fitted)
    df_out = grouped.copy()
    df_out["model_intensity"] = preds
    st.subheader("Measurements vs. model")
    st.dataframe(df_out)
