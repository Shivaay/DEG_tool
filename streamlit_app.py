# ==========================================================
# PhoenixBioInfoSys Professional Dashboard
# Streamlit Cloud Compatible UI Wrapper
# ==========================================================

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

import upload   # Your original engine

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PhoenixBioInfoSys DEG Platform",
    layout="wide",
    page_icon="🧬"
)

# ==========================================================
# UI STYLE
# ==========================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.title("🧬 PhoenixBioInfoSys Transcriptomic Intelligence Platform")
st.caption("Multi-layer biomathematical DEG interpretation environment")

st.divider()

# ==========================================================
# SESSION STORAGE FOR DOWNLOAD HUB
# ==========================================================

if "figures" not in st.session_state:
    st.session_state.figures = []

if "tables" not in st.session_state:
    st.session_state.tables = {}

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("Analysis Workflow")

page = st.sidebar.radio(
    "Navigate",
    [
        "Upload & DEG Analysis",
        "Visualization",
        "Network Biology",
        "Regulatory Biology",
        "Advanced Algorithms",
        "Interpretation Engine",
        "Download Center"
    ]
)

# ==========================================================
# PAGE 1 — DEG PIPELINE
# ==========================================================

if page == "Upload & DEG Analysis":

    st.header("📂 Upload & Differential Expression")

    # CALL ORIGINAL TOOL FUNCTION
    upload.run_deg_pipeline()


# ==========================================================
# PAGE 2 — VISUALIZATION
# ==========================================================

elif page == "Visualization":

    st.header("📊 Expression Visual Analytics")

    upload.render_volcano()

    if hasattr(upload, "render_heatmap"):
        upload.render_heatmap()

    # ======================================================
    # GO VISUALIZATION (NEW)
    # ======================================================

    if hasattr(upload, "up_en") and hasattr(upload, "down_en"):

        st.subheader("🧬 Gene Ontology Visualization")

        tabs = st.tabs(["Upregulated", "Downregulated"])

        def go_bar(df, title):
            if df.empty:
                return None
            top = df.sort_values("p_value").head(10)

            fig, ax = plt.subplots()
            ax.barh(top["name"], -np.log10(top["p_value"]))
            ax.set_xlabel("-log10(p-value)")
            ax.set_title(title)
            return fig

        def go_pie(df, title):
            if df.empty:
                return None
            top = df.sort_values("p_value").head(6)

            fig, ax = plt.subplots()
            ax.pie(top["intersection_size"], labels=top["name"], autopct="%1.1f%%")
            ax.set_title(title)
            return fig

        with tabs[0]:
            fig = go_bar(upload.up_en, "Upregulated GO")
            if fig:
                st.pyplot(fig)
                st.session_state.figures.append(("GO_Bar_Up", fig))

        with tabs[1]:
            fig = go_bar(upload.down_en, "Downregulated GO")
            if fig:
                st.pyplot(fig)
                st.session_state.figures.append(("GO_Bar_Down", fig))


# ==========================================================
# PAGE 3 — PPI NETWORK
# ==========================================================

elif page == "Network Biology":

    st.header("🔗 Protein Interaction Network")

    upload.render_ppi_network()


# ==========================================================
# PAGE 4 — miRNA & TF NETWORK
# ==========================================================

elif page == "Regulatory Biology":

    st.header("🧬 Regulatory Network Analysis")

    upload.render_mirna_network()
    upload.render_tf_network()


# ==========================================================
# PAGE 5 — ADVANCED ALGORITHMS
# ==========================================================

elif page == "Advanced Algorithms":

    st.header("⚙️ Adaptive Biomath & Machine Learning")

    upload.run_advanced_algorithms()


# ==========================================================
# PAGE 6 — INTERPRETATION ENGINE
# ==========================================================

elif page == "Interpretation Engine":

    st.header("🧠 Multi-layer Molecular Interpretation")

    upload.run_interpretation()


# ==========================================================
# PAGE 7 — DOWNLOAD HUB
# ==========================================================

elif page == "Download Center":

    st.header("📦 Export & Download Center")

    st.info("All generated figures and tables are collected here.")

    # ---------- FIGURES ----------
    for name, fig in st.session_state.figures:

        buffer = io.BytesIO()
        fig.savefig(buffer, dpi=300, bbox_inches="tight")

        st.download_button(
            f"Download {name}",
            buffer.getvalue(),
            f"{name}.png"
        )

    # ---------- TABLES ----------
    for name, table in st.session_state.tables.items():

        if isinstance(table, pd.DataFrame):

            st.download_button(
                f"Download {name}",
                table.to_csv(index=False),
                f"{name}.csv"
            )
