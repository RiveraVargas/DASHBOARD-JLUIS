import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import plotly.express as px

# -----------------------------
# Configuración de página
# -----------------------------
st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Cargar datos
# -----------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["species"] = [
        iris.target_names[i] for i in iris.target
    ]
    return df

df = load_data()

# -----------------------------
# Título
# -----------------------------
st.title("🌸 Dashboard Interactivo - Dataset Iris")
st.markdown(
    "Visualización y análisis exploratorio del dataset Iris utilizando Streamlit y Plotly."
)

# -----------------------------
# Métricas
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Muestras", len(df))

with col2:
    st.metric("Variables", len(df.columns) - 1)

with col3:
    st.metric("Especies", df["species"].nunique())

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filtros")

species_selected = st.sidebar.multiselect(
    "Seleccionar especies",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species_selected)]

numeric_cols = filtered_df.select_dtypes(
    include=np.number
).columns.tolist()

# -----------------------------
# Variables para gráficos
# -----------------------------
x_var = st.sidebar.selectbox(
    "Variable X",
    numeric_cols,
    index=0
)

y_var = st.sidebar.selectbox(
    "Variable Y",
    numeric_cols,
    index=1
)

hist_var = st.sidebar.selectbox(
    "Variable Histograma",
    numeric_cols,
    index=0
)

# -----------------------------
# Histograma
# -----------------------------
st.subheader("Distribución de Variables")

fig_hist = px.histogram(
    filtered_df,
    x=hist_var,
    color="species",
    color_discrete_sequence=px.colors.sequential.Viridis,
    nbins=20,
)

st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# Scatter Plot
# -----------------------------
st.subheader("Relación entre Variables")

fig_scatter = px.scatter(
    filtered_df,
    x=x_var,
    y=y_var,
    color="species",
    size_max=15,
    color_discrete_sequence=px.colors.sequential.Viridis,
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# Heatmap correlación
# -----------------------------
st.subheader("Matriz de Correlación")

corr = filtered_df[numeric_cols].corr()

fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis",
    aspect="auto"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# Boxplot
# -----------------------------
st.subheader("Boxplot por Especie")

box_var = st.selectbox(
    "Seleccionar variable",
    numeric_cols
)

fig_box = px.box(
    filtered_df,
    x="species",
    y=box_var,
    color="species",
    color_discrete_sequence=px.colors.sequential.Viridis
)

st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# Distribución de especies
# -----------------------------
st.subheader("Distribución de Especies")

species_count = (
    filtered_df["species"]
    .value_counts()
    .reset_index()
)

species_count.columns = ["species", "count"]

fig_bar = px.bar(
    species_count,
    x="species",
    y="count",
    color="species",
    color_discrete_sequence=px.colors.sequential.Viridis
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Tabla de datos
# -----------------------------
st.subheader("Datos Filtrados")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# -----------------------------
# Estadísticas descriptivas
# -----------------------------
st.subheader("Estadísticas Descriptivas")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)

# -----------------------------
# Correlaciones
# -----------------------------
st.subheader("Correlación entre Variables")

st.dataframe(
    corr.round(3),
    use_container_width=True
)

# -----------------------------
# Pie de página
# -----------------------------
st.divider()
st.caption(
    "Dashboard desarrollado con Streamlit, Plotly y Scikit-Learn."
)
