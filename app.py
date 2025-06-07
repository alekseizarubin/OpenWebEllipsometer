import streamlit as st
import pandas as pd
import numpy as np
from ellipsometer import Layer, ModelParams, group_measurements, predict_dataframe, fit_parameters

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

uploaded = st.file_uploader("Загрузите CSV таблицу измерений", type="csv")
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.session_state["meas_df"] = pd.concat([st.session_state["meas_df"], df_up])
    st.experimental_rerun()

df = st.session_state["meas_df"].copy()
if df.empty:
    st.stop()

grouped = group_measurements(df)
st.subheader("Сводные данные")
st.dataframe(grouped)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Параметры модели")

n_before = st.sidebar.number_input("n до плёнки", value=1.0)

with st.sidebar.expander("Тонкая плёнка"):
    n_layer = st.number_input("n плёнки", value=1.7)
    k_layer = st.number_input("k плёнки", value=0.0)
    d_layer = st.number_input("толщина, нм", value=50.0)
    opt_n_layer = st.checkbox("подгонять n", value=False)
    opt_k_layer = st.checkbox("подгонять k", value=False)
    opt_d_layer = st.checkbox("подгонять толщину", value=False)

with st.sidebar.expander("Подложка"):
    n_sub = st.number_input("n подложки", value=1.45)
    k_sub = st.number_input("k подложки", value=0.0)
    opt_n_sub = st.checkbox("подгонять n подложки", value=False)
    opt_k_sub = st.checkbox("подгонять k подложки", value=False)

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

if st.button("Запустить оптимизацию"):
    fitted, rmse = fit_parameters(grouped, params, optimise)
    st.subheader("Результаты подгонки")
    st.write(f"n плёнки: {fitted.layers[0].n:.4f}")
    st.write(f"k плёнки: {fitted.layers[0].k:.4f}")
    st.write(f"толщина плёнки [нм]: {fitted.layers[0].thickness_nm:.2f}")
    st.write(f"n подложки: {fitted.n_sub:.4f}")
    st.write(f"k подложки: {fitted.k_sub:.4f}")
    st.write(f"RMSE: {rmse:.6f}")

    preds = predict_dataframe(grouped, fitted)
    df_out = grouped.copy()
    df_out["model_intensity"] = preds
    st.subheader("Сравнение измерений и модели")
    st.dataframe(df_out)
