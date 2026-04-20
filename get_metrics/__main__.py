"""
Main script to calculate metrics. Metrics that are covered:
- Continuous Structure:
    - Continuous Structure Rate: continuous_structure_rate.py
    - Diversity-adjusted Context Progression Rate: context_progression_rate_fitted_metrics.py
    - Connectives-adjusted Context Progression Rate: context_progression_rate_fitted_metrics.py
- Cause-effect fitted metric:
    - Cause-effect-adjusted Context Progression Rate: cause_effect.py (TODO)
Linguistic fluency:
    - Fluency-Diversity Rate (DFR): fdr.py (TODO)
Lexical Diversity:
    - Type-token-adjusted Context Progression Rate: context_progression_rate_fitted_metrics.py
    - Verbs ratio-adjusted Context Progression Rate: context_progression_rate_fitted_metrics.py
"""

import subprocess


def main():
    print("\nStarting metrics computation.\n")

    # Connectives
    # all connectives
    subprocess.run(["python", "-m", "get_metrics.connectives_all"]) # only ratio
    # # contingency
    # subprocess.run(["python", "-m", "get_metrics.connectives_contingency"])
    # # expansion
    # subprocess.run(["python", "-m", "get_metrics.connectives_expansion"])

    # Lexical Diversity
    # type-token ratio
    subprocess.run(["python", "-m", "get_metrics.lexical_diversity_ttr"]) # only ratio
    # # verbs ratio
    subprocess.run(["python", "-m", "get_metrics.lexical_diversity_vr"]) # only ratio

    # FDR
    # 2-gram/ln(perplexity)
    subprocess.run(
         ["python", "-m", "get_metrics.fdr"]
    )  # runs longer due to PPL calculations

    # # Perplexity
    # # sentence ordering (original, reversed, shuffled, leave-one-out)
    subprocess.run(["python", "-m", "get_metrics.perplexity"])

    # # Continuity
    # # Shuffled sentences PPL ratio - Continuous Structure Rate (CSR)
    subprocess.run(["python", "-m", "get_metrics.continuous_structure_rate"]) 
    # DCPR (old dwar), CCPR (old cwar), TTCPR (old ttwar), VCPR (old vvar) - fit exponential curve
    subprocess.run(["python", "-m", "get_metrics.context_progression_rate_fitted_metrics"])

    # Cause-effect fitted metric
    # CE ratio based on lexicon + CECPR
    subprocess.run(["python", "-m", "get_metrics.cause_effect"])


if __name__ == "__main__":
    main()
