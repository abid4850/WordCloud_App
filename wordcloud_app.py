import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =======================
# Streamlit App Settings
# =======================
st.set_page_config(
    page_title="Data Analysis & Visualization App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis & Visualization App")
st.subheader("A professional data analysis tool by Abid Hussain")

# =======================
# Dataset Selection
# =======================
dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox("Choose a sample dataset", dataset_options)
uploaded_file = st.file_uploader("Or upload your own CSV/XLSX file", type=["csv", "xlsx"])

# =======================
# Load Dataset
# =======================
def load_dataset(file, sample_name):
    try:
        if file is not None:
            if file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding="latin1")
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            st.success("✅ Custom dataset uploaded successfully!")
        else:
            df = sns.load_dataset(sample_name)
            st.success(f"✅ Loaded '{sample_name}' dataset from Seaborn")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        df = pd.DataFrame()
    return df

df = load_dataset(uploaded_file, selected_dataset)
if df.empty:
    st.stop()

# =======================
# Dataset Overview
# =======================
st.subheader("Dataset Preview")
st.dataframe(df)

st.write(f"**Number of rows:** {df.shape[0]}")
st.write(f"**Number of columns:** {df.shape[1]}")

st.subheader("Column Names & Data Types")
st.write(df.dtypes)

st.subheader("Missing Values")
missing = df.replace("", np.nan).isnull().sum()
if missing.sum() > 0:
    st.dataframe(missing[missing > 0].sort_values(ascending=False))
else:
    st.info("No missing values in this dataset.")

st.subheader("Summary Statistics")
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    st.dataframe(numeric_df.describe())
else:
    st.info("No numeric columns to display summary statistics.")

# =======================
# Pairplot
# =======================
st.subheader("Pairplot")
try:
    hue_column = st.selectbox("Select a column for hue (optional)", [None] + df.columns.tolist())
    hue_column = None if hue_column == "None" else hue_column
    pairplot_fig = sns.pairplot(df, hue=hue_column)
    st.pyplot(pairplot_fig)
except Exception as e:
    st.warning(f"Pairplot could not be generated: {e}")

# =======================
# Correlation Heatmap
# =======================
st.subheader("Correlation Heatmap")
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            hoverongaps=False
        )
    )
    heatmap_fig.update_layout(width=700, height=600)
    st.plotly_chart(heatmap_fig)
else:
    st.info("No numeric columns for correlation heatmap.")

# =======================
# Interactive Column Plot
# =======================
st.subheader("Interactive Column Plot")
if numeric_df.shape[1] > 0:
    numeric_columns = numeric_df.columns.tolist()
    x_col = st.selectbox("X-axis", numeric_columns)
    y_col = st.selectbox("Y-axis", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
    plot_type = st.selectbox("Plot Type", ["Line", "Bar", "Scatter", "Histogram", "KDE"])

    plt.figure(figsize=(8, 5))
    try:
        if plot_type == "Line":
            plt.plot(df[x_col], df[y_col], marker='o')
        elif plot_type == "Bar":
            plt.bar(df[x_col], df[y_col])
        elif plot_type == "Scatter":
            plt.scatter(df[x_col], df[y_col])
        elif plot_type == "Histogram":
            plt.hist(df[x_col].dropna(), bins=20)
        elif plot_type == "KDE":
            sns.kdeplot(df[x_col].dropna().astype(float), fill=True)
        plt.xlabel(x_col)
        plt.ylabel(y_col if plot_type != "Histogram" else "Frequency")
        plt.title(f"{plot_type} Plot of {x_col} vs {y_col}" if plot_type != "Histogram" else f"{plot_type} of {x_col}")
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not generate plot: {e}")
else:
    st.info("No numeric columns available for plotting.")

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("Made with ❤️ by **Abid Hussain**")
