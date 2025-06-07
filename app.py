import streamlit as st
import pandas as pd
import numpy as np
import refnx.dataset as rds
from refnx.reflect import SLD, Layer, Structure, Objective, ReflectModel
from scipy.optimize import DifferentialEvolution

st.title("DIY-Ellipsometry fitter")

uploaded = st.file_uploader("Загрузите CSV (λ, Ψ, Δ)", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    lam = data.iloc[:,0].to_numpy()*1e-9        # → м
    psi = np.deg2rad(data.iloc[:,1])
    delta = np.deg2rad(data.iloc[:,2])
    # --- модель слоя: подложка стекло / плёнка / воздух ---
    n_sub = st.number_input("n подложки", 1.45)
    d_layer = st.number_input("Толщина плёнки, нм", 50.0)
    n_layer = st.number_input("n плёнки @ 550 нм", 1.70)
    k_layer = st.number_input("k плёнки @ 550 нм", 0.05)

    # Собираем структуру refnx
    sub = SLD(n_sub)(0)
    film = SLD(complex(n_layer, k_layer))(d_layer)
    air = SLD(1)(0)
    structure = Structure([sub, film, air])

    # Строим модель Ψ/Δ
    model = ReflectModel(structure, bkg=0, dq=0)
    # ... здесь можно добавить Objective и оптимизацию ...

    st.subheader("Предпросмотр модели")
    psi_calc, delta_calc = model(lam, output="psi_delta")
    df_out = pd.DataFrame({
        "λ (nm)": lam*1e9,
        "Ψ експ [°]": np.rad2deg(psi),
        "Ψ calc [°]": np.rad2deg(psi_calc),
        "Δ експ [°]": np.rad2deg(delta),
        "Δ calc [°]": np.rad2deg(delta_calc),
    })
    st.line_chart(df_out.set_index("λ (nm)"))
