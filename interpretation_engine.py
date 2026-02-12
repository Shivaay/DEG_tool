# ==========================================================
# INTERPRETATION ENGINE
# ==========================================================

import json

SAFE_PROMPT = """
Role:
You are an AI scientific interpretation assistant specialized in transcriptomics and systems biology.

Rules:
• Use ONLY provided structured data
• Avoid causality
• Avoid clinical diagnosis
• Provide hypothesis-level interpretation
• State limitations if evidence is weak

Output Sections:

1. Summary of Key Findings
2. Multi-Layer Evidence Integration
3. Candidate Regulatory Mechanisms
4. Hypotheses for Experimental Validation
5. Limitations and Analytical Context
"""


# ----------------------------------------------------------
# BUILD STRUCTURED DATA
# ----------------------------------------------------------

def build_structured_results(
    up_genes=None,
    down_genes=None,
    enrichment_dict=None,
    hub_genes=None,
    mirna_df=None,
    tf_df=None
):

    structured = {
        "ranked_genes": {
            "upregulated": up_genes[:20] if up_genes else [],
            "downregulated": down_genes[:20] if down_genes else []
        },

        "ppi_summary": {
            "hub_genes": hub_genes if hub_genes else []
        },

        "enrichment": enrichment_dict if enrichment_dict else {},

        "mirna_network": (
            mirna_df.head(20).to_dict("records")
            if mirna_df is not None else []
        ),

        "tf_network": (
            tf_df.head(20).to_dict("records")
            if tf_df is not None else []
        )
    }

    return structured


# ----------------------------------------------------------
# GENERATE INTERPRETATION
# ----------------------------------------------------------

def generate_interpretation(client, structured_data):

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SAFE_PROMPT},
            {"role": "user", "content": json.dumps(structured_data)}
        ]
    )

    return response.choices[0].message.content
