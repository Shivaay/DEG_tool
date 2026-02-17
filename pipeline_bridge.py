# =========================================================
# REAL PIPELINE BRIDGE
# =========================================================

from biomath_layer import run_biomath_layer

from interpretation_engine import (
    InterpretationInput,
    InterpretationEngine
)


class BioPipelineBridge:


    def run_pipeline(

            self,

            deg_df,

            gene_col,

            logfc_col,

            pval_col,

            ppi_df,

            hub_df

    ):


        biomath_df, metrics = run_biomath_layer(

            deg_df,

            gene_col,

            logfc_col,

            pval_col,

            ppi_df

        )


        interpretation_input = InterpretationInput(

            deg_table=biomath_df,

            biomath_metrics=metrics,

            hub_genes=hub_df

        )


        engine = InterpretationEngine(

            interpretation_input

        )


        report = engine.generate()


        return {

            "biomath_table": biomath_df,

            "metrics": metrics,

            "report": report

        }
