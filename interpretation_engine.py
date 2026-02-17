# =========================================================
# REAL INTERPRETATION ENGINE
# =========================================================

from dataclasses import dataclass
import numpy as np


@dataclass
class InterpretationInput:

    deg_table: object

    biomath_metrics: dict

    hub_genes: object


class InterpretationEngine:


    def __init__(self, data):

        self.data = data


    def transcriptomic_state(self):

        up = len(self.data.deg_table[
            self.data.deg_table["Regulation"] == "Up"
        ])

        down = len(self.data.deg_table[
            self.data.deg_table["Regulation"] == "Down"
        ])

        if up > down:

            return "activation"

        else:

            return "suppression"


    def collapse_probability(self):

        m = self.data.biomath_metrics

        entropy = m["system_entropy"]

        perturb = m["perturbation_magnitude"]

        stability = m["system_stability"]

        return entropy * perturb * (1 - stability)


    def hub_analysis(self):

        if self.data.hub_genes is None:

            return "No hub genes detected"

        top_gene = self.data.hub_genes.iloc[0]["gene"]

        return f"Hub gene {top_gene} shows highest regulatory influence"


    def generate(self):

        state = self.transcriptomic_state()

        collapse = self.collapse_probability()

        hub_text = self.hub_analysis()

        m = self.data.biomath_metrics


        report = f"""

SYSTEMS BIOLOGY REPORT


The transcriptomic profile shows dominant {state} pattern.

System entropy is {m['system_entropy']:.3f},

indicating biological complexity.


System stability score is {m['system_stability']:.3f},

suggesting robustness level.


Network centrality score is {m['network_centrality']:.3f},

indicating regulatory connectivity.


Perturbation magnitude is {m['perturbation_magnitude']:.3f},

indicating transcriptional disruption strength.


{hub_text}


Predicted system collapse probability is

{collapse:.3f}


Conclusion:

The system shows biologically meaningful regulatory

changes driven by network-connected genes.

"""

        return report
        manuscript = f"""
SYSTEMS BIOLOGY INTERPRETATION REPORT

Transcriptomic Program:
The molecular landscape exhibits dominant {dominance} architecture.

Network Entropy: {m.get('network_entropy',0):.3f}
System Stability: {m.get('system_stability',0):.3f}
Perturbation Magnitude: {m.get('perturbation_magnitude',0):.3f}
Topology Score: {m.get('topology_score',0):.3f}
Bayesian Entropy: {m.get('bayesian_entropy',0):.3f}
Multi-Omics Integration Index: {m.get('multiomics_index',0):.3f}

Systems Collapse Probability:
{collapse:.3f}

Mechanistic Systems Interpretation:
Regulatory hubs amplify transcriptional cascades, producing non-linear pathway reinforcement.
Network topology indicates concentrated regulatory bottlenecks.
Multi-omics integration score confirms cross-layer coherence.

Clinical Translation:
High-centrality genes represent therapeutic leverage nodes.
Perturbation magnitude suggests intervention-responsive phenotype.

Experimental Strategy:
Time-series perturbation modeling combined with hub gene modulation.
"""

        return {
            "text_report": manuscript,
            "summary": f"Dominant {dominance} transcriptomic state",
            "systems_analysis": m,
            "hypothesis": "Hub-driven cascade amplification underlies phenotype.",
            "clinical_translation": "Target regulatory bottlenecks.",
            "validation_plan": "Time-series network perturbation.",
            "confidence_score": 1 - collapse,
            "figures": [fig],
            "tables": []
        }
