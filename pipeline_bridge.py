# ============================================================
# PIPELINE BRIDGE LAYER
# Connects:
# Traditional DEG Tool
# Biomath Layer
# Interpretation Engine
# ============================================================

import pandas as pd
import numpy as np

from biomath_layer import run_biomath_layer
from interpretation_engine import InterpretationInput, InterpretationEngine
from interpreter_layer import run_interpreter_layer


# ============================================================
# CONFIGURATION CLASS
# ============================================================

class PipelineConfig:
    def __init__(self):

        self.required_deg_columns = [
            "Gene",
            "logFC",
            "pvalue",
            "Regulation"
        ]

        self.optional_columns_with_fallback = {
            "baseMean": lambda df: df["logFC"].abs() + 1
        }


# ============================================================
# VALIDATION + NORMALIZATION
# ============================================================

def normalize_deg_table(deg_df, gene_col, logfc_col, pval_col):

    df = deg_df.copy()

    # Standardize column names internally
    df.rename(columns={
        gene_col: "Gene",
        logfc_col: "logFC",
        pval_col: "pvalue"
    }, inplace=True)

    # Ensure Regulation exists
    if "Regulation" not in df.columns:
        raise ValueError("Regulation column missing from DEG results")

    return df


# ============================================================
# FALLBACK COLUMN HANDLER
# ============================================================

def ensure_optional_columns(df, config: PipelineConfig):

    for col, generator in config.optional_columns_with_fallback.items():

        if col not in df.columns:
            df[col] = generator(df)

    return df


# ============================================================
# PPI EDGE STANDARDIZATION
# ============================================================

def normalize_ppi_edges(ppi_df):

    if ppi_df is None or ppi_df.empty:
        return None

    expected = ["preferredName_A", "preferredName_B"]

    if all(c in ppi_df.columns for c in expected):
        return ppi_df[expected]

    # Attempt fallback column guessing
    return ppi_df.iloc[:, :2].rename(columns={
        ppi_df.columns[0]: "preferredName_A",
        ppi_df.columns[1]: "preferredName_B"
    })


# ============================================================
# MAIN PIPELINE CONTROLLER
# ============================================================

class BioPipelineBridge:

    def __init__(self):

        self.config = PipelineConfig()

    # --------------------------------------------------------
    # TRADITIONAL OUTPUT PROCESSING
    # --------------------------------------------------------
    def process_traditional_output(
        self,
        deg_df,
        gene_col,
        logfc_col,
        pval_col
    ):

        df = normalize_deg_table(deg_df, gene_col, logfc_col, pval_col)

        df = ensure_optional_columns(df, self.config)

        return df


    # --------------------------------------------------------
    # BIOMATH EXECUTION
    # --------------------------------------------------------
    def run_biomath(self, deg_df, ppi_df):

        if ppi_df is None or ppi_df.empty:
            return None

        ppi_edges = normalize_ppi_edges(ppi_df)

        try:
            biomath_df = run_biomath_layer(deg_df, ppi_edges)
            return biomath_df
        except Exception as e:
            print("Biomath Layer Error:", str(e))
            return None


    # --------------------------------------------------------
    # INTERPRETATION EXECUTION
    # --------------------------------------------------------
    def run_interpretation(
        self,
        deg_df,
        up_df,
        down_df,
        hub_df,
        enrichment_up,
        enrichment_down,
        mirna_df,
        tf_df,
        biomath_df,
        ppi_df
    ):

        try:

            interpreter_results = None

            if biomath_df is not None and ppi_df is not None and not ppi_df.empty:

                ppi_edges = normalize_ppi_edges(ppi_df)

                interpreter_results = run_interpreter_layer(
                    biomath_df,
                    ppi_edges
                )

            input_data = InterpretationInput(
                deg_table=deg_df,
                up_genes=up_df,
                down_genes=down_df,
                hub_genes=hub_df,
                enrichment_up=enrichment_up,
                enrichment_down=enrichment_down,
                mirna_df=mirna_df,
                tf_df=tf_df,
                bayesian_confidence=None,
                adaptive_threshold=None
            )

            engine = InterpretationEngine(input_data)

            return {
                "text_report": engine.generate_report(),
                "interpreter_results": interpreter_results
            }

        except Exception as e:
            print("Interpretation Error:", str(e))
            return None


    # --------------------------------------------------------
    # COMPLETE PIPELINE RUNNER
    # --------------------------------------------------------
    def run_full_pipeline(
        self,
        deg_df,
        gene_col,
        logfc_col,
        pval_col,
        ppi_df,
        enrichment_up,
        enrichment_down,
        mirna_df,
        tf_df,
        hub_df
    ):

        # STEP 1: Normalize Traditional Output
        normalized_deg = self.process_traditional_output(
            deg_df,
            gene_col,
            logfc_col,
            pval_col
        )

        # STEP 2: Separate Up / Down
        up_df = normalized_deg[normalized_deg["Regulation"] == "Up"]
        down_df = normalized_deg[normalized_deg["Regulation"] == "Down"]

        # STEP 3: Biomath
        biomath_df = self.run_biomath(normalized_deg, ppi_df)

        # STEP 4: Interpretation
        interpretation = self.run_interpretation(
            normalized_deg,
            up_df,
            down_df,
            hub_df,
            enrichment_up,
            enrichment_down,
            mirna_df,
            tf_df,
            biomath_df,
            ppi_df
        )

        return {
            "traditional_deg": normalized_deg,
            "biomath_deg": biomath_df,
            "interpretation": interpretation
        }
