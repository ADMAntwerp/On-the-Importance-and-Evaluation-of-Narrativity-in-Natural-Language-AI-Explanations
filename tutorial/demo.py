"""
XAI Metrics Comparison: Description vs Narrative

Metrics computed:
- FDR (Fluency-Diversity Rate)
- Continuous Structure Rate:  CSR = ((PPL_shuffled - PPL_original) / PPL_original)
- DCPR, CECPR, CCPR, TTCPR, VCPR  (use r from fitting y(x) = A * r^x + C
  to cumulative perplexities; NOT the CSR ratio)

Usage:
    python scripts/compare_xai_texts.py
"""

import math

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModelForCausalLM

from get_metrics.lexical_diversity_ttr import compute_ttr
from get_metrics.lexical_diversity_vr import compute_verb_ratio
from get_metrics.connectives_all import compute_connectives_ratio
from get_metrics.cause_effect import compute_cause_effect_ratio, load_cause_effect_markers
from utils.connectives_lex import connectives
from utils.perplexity_utils import (
    calculate_fdr,
    calculate_ppl,
    calculate_perplexity_change_ratio,
    calculate_ppl_cumulative,
    calculate_distinct_bigram_ratio,
    shuffle_sentences,
    split_sentences,
)


def fit_decay_rate(cumulative_ppls):
    """Fit y(x) = A * exp(-b*x) + C to cumulative perplexities and return
    (r, A, C, fit_quality) where r = exp(-b) is the geometric decay rate
    used by the Context Progression Rate metrics in the paper.

    Mirrors the bounding strategy of
    utils.perplexity_utils.compute_cumulative_ppl_and_dcpr so the demo and
    the dataset-level pipeline agree.
    """
    y_values = np.asarray(cumulative_ppls, dtype=float)
    x_values = np.arange(1, len(y_values) + 1, dtype=float)

    if len(y_values) < 2:
        return None, None, None, None

    def exp_offset(x, a, b, c):
        return a * np.exp(-b * x) + c

    min_ppl = float(np.min(y_values))
    max_ppl = float(np.max(y_values))
    is_short = len(x_values) <= 3

    try:
        if is_short:
            estimated_c = max(0.1, min_ppl - 1.0)
            c_lower, c_upper = estimated_c - 0.05, estimated_c + 0.05
        else:
            c_lower = 0.1
            c_upper = min_ppl - 0.01

        bounds = ([0.1, 0.001, c_lower], [np.inf, 10.0, c_upper])
        p0 = [max(max_ppl - min_ppl, 0.1), 0.5, max(min_ppl - 1.0, c_lower)]

        popt, _ = curve_fit(
            exp_offset, x_values, y_values, p0=p0, bounds=bounds, maxfev=10000
        )
        a_opt, b_opt, c_opt = popt
        r_val = float(np.exp(-b_opt))

        y_fit = exp_offset(x_values, a_opt, b_opt, c_opt)
        residuals = y_values - y_fit
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return r_val, float(a_opt), float(c_opt), {"R2": r_squared, "RMSE": rmse}
    except Exception as exc:
        print(f"  [warn] exponential fit failed: {exc}")
        return None, None, None, None


def compute_csr_metrics(text, model, tokenizer, cause_effect_markers):
    """Compute CSR and the Context Progression Rate metrics for a single text.

    CSR uses the sentence-shuffle perplexity ratio. The CPR-style metrics
    (DCPR, TTCPR, VCPR, CCPR, CECPR) use r from fitting y(x) = A*r^x + C
    to cumulative perplexities, NOT the CSR value.
    """
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return {"error": "Text must have at least 2 sentences"}

    ppl_original = calculate_ppl(text, model, tokenizer)
    shuffled_text = shuffle_sentences(text)
    ppl_shuffled = calculate_ppl(shuffled_text, model, tokenizer)
    csr_val = calculate_perplexity_change_ratio(ppl_original, ppl_shuffled)

    # r for CPR metrics: fit shifted exponential to cumulative perplexities
    cumulative = calculate_ppl_cumulative(text, model, tokenizer)
    cumulative_ppls = [step["ppl"] for step in cumulative]
    r_val, a_fit, c_fit, fit_quality = fit_decay_rate(cumulative_ppls)

    # Compute diversity metrics
    distinct_2gram_ratio = calculate_distinct_bigram_ratio(text)
    ttr_val = compute_ttr(text)
    vr_val = compute_verb_ratio(text)
    cd_val = compute_connectives_ratio(text, connectives)
    ce_val = compute_cause_effect_ratio(text, cause_effect_markers)

    results = {
        "CSR": round(csr_val, 6) if csr_val is not None else None,
        "PPL_base": round(ppl_original, 2),
        "PPL_delta": round(ppl_shuffled - ppl_original, 2),
        "r_fit": round(r_val, 6) if r_val is not None else None,
        "A_fit": round(a_fit, 4) if a_fit is not None else None,
        "C_fit": round(c_fit, 4) if c_fit is not None else None,
        "fit_R2": round(fit_quality["R2"], 4) if fit_quality else None,
        "fit_RMSE": round(fit_quality["RMSE"], 4) if fit_quality else None,
        "CE": round(ce_val, 4),
    }

    if r_val is not None:
        results["DCPR"] = round(r_val / (distinct_2gram_ratio ** 2), 4) if distinct_2gram_ratio > 0 else None
        results["TTCPR"] = round(r_val / (ttr_val ** 2), 4) if ttr_val > 0 else None
        results["VCPR"] = round(r_val / (vr_val ** 2), 4) if vr_val > 0 else None
        results["CCPR"] = round(r_val / (cd_val ** 2), 4) if cd_val > 0 else None
        results["CECPR"] = round(r_val / (ce_val ** 2), 4) if ce_val > 0 else None
    else:
        results["DCPR"] = None
        results["TTCPR"] = None
        results["VCPR"] = None
        results["CCPR"] = None
        results["CECPR"] = None

    return results


def compute_fdr_metric(text, model, tokenizer):
    """Compute FDR for a single text."""
    ppl = calculate_ppl(text, model, tokenizer)
    fdr = calculate_fdr(text, ppl)
    return round(fdr, 4), round(ppl, 2)


def compare_texts(texts: dict, model, tokenizer):
    """Compare XAI metrics across multiple texts.

    Args:
        texts: dict mapping label -> text string
    """
    cause_effect_markers = load_cause_effect_markers()

    metric_values = {}
    for label, text in texts.items():
        text = " ".join(text.split())
        print(f"Computing metrics for {label}...")
        fdr, _ = compute_fdr_metric(text, model, tokenizer)
        csr = compute_csr_metrics(text, model, tokenizer, cause_effect_markers)
        metric_values[label] = {"FDR": fdr, **csr}

    metrics_order = ["FDR", "CSR", "DCPR", "CECPR", "CCPR", "TTCPR", "VCPR"]
    metrics_direction = {
        "FDR": "↑", "CSR": "↑",
        "DCPR": "↓", "CECPR": "↓", "CCPR": "↓", "TTCPR": "↓", "VCPR": "↓",
    }

    comparison_data = []
    for metric_name in metrics_order:
        row = {"Metric": f"{metric_name} {metrics_direction[metric_name]}"}
        for label in texts:
            row[label] = metric_values[label].get(metric_name)
        comparison_data.append(row)

        if metric_name == "CSR":
            nominal_row = {"Metric": "  CSR_nominal (PPL_base → +Δ)"}
            for label in texts:
                base = metric_values[label].get("PPL_base")
                delta = metric_values[label].get("PPL_delta")
                nominal_row[label] = f"{base} → +{delta}" if base is not None else None
            comparison_data.append(nominal_row)

    return pd.DataFrame(comparison_data)


def compute_nlp_metrics(texts: dict, model, tokenizer):
    """Compute standard NLP metrics for each text.

    Args:
        texts: dict mapping label -> text string
    Returns:
        pd.DataFrame with one row per metric and one column per text
    """
    nlp_metrics_order = ["PPL", "Dist2", "TTR", "VR", "CD"]
    nlp_direction = {"PPL": "↓", "Dist2": "↑", "TTR": "↑", "VR": "↑", "CD": "↑"}

    scores = {label: {} for label in texts}
    for label, text in texts.items():
        text = " ".join(text.split())
        ppl = calculate_ppl(text, model, tokenizer)
        scores[label]["PPL"] = round(ppl, 2)
        scores[label]["Dist2"] = round(calculate_distinct_bigram_ratio(text), 4)
        scores[label]["TTR"] = round(compute_ttr(text), 4)
        scores[label]["VR"] = round(compute_verb_ratio(text), 4)
        scores[label]["CD"] = round(compute_connectives_ratio(text, connectives), 4)

    rows = []
    for metric in nlp_metrics_order:
        row = {"Metric": f"{metric} {nlp_direction[metric]}"}
        for label in texts:
            row[label] = scores[label][metric]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_walkthrough(text, model, tokenizer, cause_effect_markers):
    """Collect every intermediate value used by each metric for one text."""
    text = " ".join(text.split())
    sentences = split_sentences(text)
    shuffled_text = shuffle_sentences(text)

    ppl_original = calculate_ppl(text, model, tokenizer)
    ppl_shuffled = calculate_ppl(shuffled_text, model, tokenizer)
    csr = calculate_perplexity_change_ratio(ppl_original, ppl_shuffled)

    cumulative = calculate_ppl_cumulative(text, model, tokenizer)
    cumulative_ppls = [step["ppl"] for step in cumulative]
    r_val, a_fit, c_fit, fit_quality = fit_decay_rate(cumulative_ppls)

    dist2 = calculate_distinct_bigram_ratio(text)
    ttr = compute_ttr(text)
    vr = compute_verb_ratio(text)
    cd = compute_connectives_ratio(text, connectives)
    ce = compute_cause_effect_ratio(text, cause_effect_markers)

    fdr = calculate_fdr(text, ppl_original)

    return {
        "n_sentences": len(sentences),
        "shuffled_text": shuffled_text,
        "PPL_original": ppl_original,
        "PPL_shuffled": ppl_shuffled,
        "CSR": csr,
        "cumulative_ppls": cumulative_ppls,
        "r_fit": r_val,
        "A_fit": a_fit,
        "C_fit": c_fit,
        "fit_quality": fit_quality,
        "Dist2": dist2,
        "TTR": ttr,
        "VR": vr,
        "CD": cd,
        "CE": ce,
        "FDR": fdr,
    }


def _fmt(x, nd=4):
    return "None" if x is None else f"{x:.{nd}f}"


def _cpr_line(name, r_val, div, div_name):
    """Render one CPR-style line: metric = r / div^2, fully substituted.

    r is the decay rate from fitting y(x) = A * r^x + C to cumulative
    perplexities (NOT the CSR shuffle ratio).
    """
    if r_val is None:
        return f"    {name:<6}= r / {div_name}^2                                     (undefined: r fit failed)"
    if div is None or div <= 0:
        return f"    {name:<6}= r / {div_name}^2                                     (undefined: {div_name} = {_fmt(div)})"
    div_sq = div ** 2
    val = r_val / div_sq
    return (
        f"    {name:<6}= r / {div_name}^2 "
        f"= {_fmt(r_val)} / ({_fmt(div)})^2 "
        f"= {_fmt(r_val)} / {_fmt(div_sq)} "
        f"= {_fmt(val)}"
    )


def print_walkthrough(texts: dict, model, tokenizer):
    """Print a step-by-step calculation of every metric for every text."""
    cause_effect_markers = load_cause_effect_markers()

    for label, text in texts.items():
        w = compute_walkthrough(text, model, tokenizer, cause_effect_markers)

        print("\n" + "=" * 70)
        print(f"DETAILED WALKTHROUGH: {label}")
        print("=" * 70)
        print(f"Sentences detected: {w['n_sentences']}")
        print(f"Shuffled (seed=42): {w['shuffled_text'][:140]}"
              + ("..." if len(w["shuffled_text"]) > 140 else ""))

        print("\n[1] Perplexity (PPL = exp(cross-entropy loss))")
        print(f"    PPL_original  = {_fmt(w['PPL_original'], 4)}")
        print(f"    PPL_shuffled  = {_fmt(w['PPL_shuffled'], 4)}")

        print("\n[2] Continuous Structure Rate  CSR = (PPL_shuffled - PPL_original) / PPL_original")
        if w["CSR"] is None:
            print("    CSR = undefined (PPL_original is 0 or None)")
        else:
            delta = w["PPL_shuffled"] - w["PPL_original"]
            print(
                f"    CSR = ({_fmt(w['PPL_shuffled'])} - {_fmt(w['PPL_original'])}) / {_fmt(w['PPL_original'])} "
                f"= {_fmt(delta)} / {_fmt(w['PPL_original'])} "
                f"= {_fmt(w['CSR'], 6)}"
            )

        print("\n[3] Diversity / structural measures")
        print(f"    Dist-2 (distinct bigram ratio) = {_fmt(w['Dist2'])}")
        print(f"    TTR    (type-token ratio)      = {_fmt(w['TTR'])}")
        print(f"    VR     (verb ratio)            = {_fmt(w['VR'])}")
        print(f"    CD     (connectives density)   = {_fmt(w['CD'])}")
        print(f"    CE     (cause-effect ratio)    = {_fmt(w['CE'])}")

        print("\n[4] Fluency-Diversity Rate  FDR = (Dist-2)^2 / ln(PPL_original)")
        if w["PPL_original"] is None or w["PPL_original"] <= 0:
            print("    FDR = undefined (PPL_original not positive)")
        else:
            dist2_sq = w["Dist2"] ** 2
            ln_ppl = math.log(w["PPL_original"])
            print(
                f"    FDR = ({_fmt(w['Dist2'])})^2 / ln({_fmt(w['PPL_original'])}) "
                f"= {_fmt(dist2_sq)} / {_fmt(ln_ppl)} "
                f"= {_fmt(w['FDR'])}"
            )

        print("\n[5] Cumulative-PPL exponential fit  y(x) = A * r^x + C")
        if w["cumulative_ppls"]:
            print(
                "    cumulative PPLs (1..n) = ["
                + ", ".join(_fmt(p, 3) for p in w["cumulative_ppls"])
                + "]"
            )
        if w["r_fit"] is None:
            print("    fit failed -> r is undefined; CPR metrics will be None")
        else:
            print(
                f"    A = {_fmt(w['A_fit'])} | r = {_fmt(w['r_fit'], 6)} | "
                f"C = {_fmt(w['C_fit'])}"
            )
            if w["fit_quality"]:
                print(
                    f"    fit quality: R^2 = {_fmt(w['fit_quality']['R2'])} | "
                    f"RMSE = {_fmt(w['fit_quality']['RMSE'])}"
                )

        print("\n[6] Context Progression Rates  CPR_x = r / x^2  (r from [5], NOT CSR)")
        print(_cpr_line("DCPR",  w["r_fit"], w["Dist2"], "Dist-2"))
        print(_cpr_line("TTCPR", w["r_fit"], w["TTR"],   "TTR"))
        print(_cpr_line("VCPR",  w["r_fit"], w["VR"],    "VR"))
        print(_cpr_line("CCPR",  w["r_fit"], w["CD"],    "CD"))
        print(_cpr_line("CECPR", w["r_fit"], w["CE"],    "CE"))


def load_model(model_name="meta-llama/Llama-3.1-8B"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


if __name__ == "__main__":
    xai_description = """
    The model predicts Good Credit Risk. The most important feature is Credit amount (3,632 EUR), with a SHAP contribution of +0.13. The second most important feature is Age (22), with a SHAP contribution of −0.12. The third most important feature is Housing (rent), with a SHAP contribution of −0.04. The fourth most important feature is Purpose (car), with a SHAP contribution of −0.03. The fifth most important feature is Sex (female), with a SHAP contribution of −0.03.
    """

    xai_narrative = """
    The model predicts Good Credit Risk for this applicant. It starts by looking at the requested credit amount of 3,632 EUR, which on its own pushes the decision toward bad credit risk. Against this starting point, the applicant's age of 22 acts as the strongest counterweight and pulls the assessment back toward good credit risk. Building on that shift, the applicant's status as a renter further softens the initial financial concern. In turn, the car loan purpose and the recorded female sex add smaller effects in the same favourable direction. As a result, these combined factors outweigh the earlier negative signal from the credit amount, and the model therefore classifies the applicant as a Good Credit Risk.
    """

    xai_strange = """
    The model made a prediction based on the data. Therefore, the outcome is what it is, because the model made a prediction. As a result, the output follows the input, since the feature matters. Feature X is important, feature Y is less important. Consequently, the score reflects the value, so the result is the outcome. Thus, the model works, and the final decision stands.
    """
    texts = {
        "XAI Description": xai_description,
        "XAI Narrative": xai_narrative,
    }

    texts_extended = {
        "XAI Description": xai_description,
        "XAI Narrative": xai_narrative,
        "Strange": xai_strange,
    }

    model, tokenizer = load_model()
    df = compare_texts(texts, model, tokenizer)

    print("\n" + "=" * 70)
    print("METRICS COMPARISON: XAI Description vs XAI Narrative vs Strange")
    print("=" * 70)
    print(df.to_string(index=False))

    df_nlp = compute_nlp_metrics(texts_extended, model, tokenizer)

    print("\n" + "=" * 70)
    print("STANDARD NLP METRICS")
    print("=" * 70)
    print(df_nlp.to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-METRIC CALCULATION WALKTHROUGH")
    print("=" * 70)
    print_walkthrough(texts_extended, model, tokenizer)


# ======================================================================
# METRICS COMPARISON: XAI Description vs XAI Narrative vs Strange
# ======================================================================
#                        Metric XAI Description XAI Narrative
#                         FDR ↑          0.2487        0.3134
#                         CSR ↑        0.394379      0.515356
#   CSR_nominal (PPL_base → +Δ)    4.65 → +1.84  15.1 → +7.78
#                        DCPR ↓          0.3348         0.178
#                       CECPR ↓            None      222.5684
#                        CCPR ↓         25.0972       16.5547
#                       TTCPR ↓          0.8003        0.3658
#                        VCPR ↓         17.4286       13.9105

# ======================================================================
# STANDARD NLP METRICS
# ======================================================================
#  Metric  XAI Description  XAI Narrative  Strange
#   PPL ↓           4.6500        15.1000  17.3700
# Dist2 ↑           0.6184         0.9224   0.9365
#   TTR ↑           0.4000         0.6435   0.5781
#    VR ↑           0.0857         0.1043   0.1875
#    CD ↑           0.0714         0.0957   0.1250

# ======================================================================
# PER-METRIC CALCULATION WALKTHROUGH
# ======================================================================

# ======================================================================
# DETAILED WALKTHROUGH: XAI Description
# ======================================================================
# Sentences detected: 6
# Shuffled (seed=42): The third most important feature is Housing (rent), with a SHAP contribution of −0.04. The most important feature is Credit amount (3,632 EU...

# [1] Perplexity (PPL = exp(cross-entropy loss))
#     PPL_original  = 4.6531
#     PPL_shuffled  = 6.4882

# [2] Continuous Structure Rate  CSR = (PPL_shuffled - PPL_original) / PPL_original
#     CSR = (6.4882 - 4.6531) / 4.6531 = 1.8351 / 4.6531 = 0.394379

# [3] Diversity / structural measures
#     Dist-2 (distinct bigram ratio) = 0.6184
#     TTR    (type-token ratio)      = 0.4000
#     VR     (verb ratio)            = 0.0857
#     CD     (connectives density)   = 0.0714
#     CE     (cause-effect ratio)    = 0.0000

# [4] Fluency-Diversity Rate  FDR = (Dist-2)^2 / ln(PPL_original)
#     FDR = (0.6184)^2 / ln(4.6531) = 0.3824 / 1.5375 = 0.2487

# [5] Cumulative-PPL exponential fit  y(x) = A * r^x + C
#     cumulative PPLs (1..n) = [491.329, 65.444, 17.754, 9.271, 6.268, 4.653]
#     A = 3800.0583 | r = 0.128047 | C = 4.6431
#     fit quality: R^2 = 0.9998 | RMSE = 2.7036

# [6] Context Progression Rates  CPR_x = r / x^2  (r from [5], NOT CSR)
#     DCPR  = r / Dist-2^2 = 0.1280 / (0.6184)^2 = 0.1280 / 0.3824 = 0.3348
#     TTCPR = r / TTR^2 = 0.1280 / (0.4000)^2 = 0.1280 / 0.1600 = 0.8003
#     VCPR  = r / VR^2 = 0.1280 / (0.0857)^2 = 0.1280 / 0.0073 = 17.4286
#     CCPR  = r / CD^2 = 0.1280 / (0.0714)^2 = 0.1280 / 0.0051 = 25.0972
#     CECPR = r / CE^2                                     (undefined: CE = 0.0000)

# ======================================================================
# DETAILED WALKTHROUGH: XAI Narrative
# ======================================================================
# Sentences detected: 6
# Shuffled (seed=42): Building on that shift, the applicant's status as a renter further softens the initial financial concern. It starts by looking at the reques...

# [1] Perplexity (PPL = exp(cross-entropy loss))
#     PPL_original  = 15.1021
#     PPL_shuffled  = 22.8850

# [2] Continuous Structure Rate  CSR = (PPL_shuffled - PPL_original) / PPL_original
#     CSR = (22.8850 - 15.1021) / 15.1021 = 7.7829 / 15.1021 = 0.515356

# [3] Diversity / structural measures
#     Dist-2 (distinct bigram ratio) = 0.9224
#     TTR    (type-token ratio)      = 0.6435
#     VR     (verb ratio)            = 0.1043
#     CD     (connectives density)   = 0.0957
#     CE     (cause-effect ratio)    = 0.0261

# [4] Fluency-Diversity Rate  FDR = (Dist-2)^2 / ln(PPL_original)
#     FDR = (0.9224)^2 / ln(15.1021) = 0.8508 / 2.7148 = 0.3134

# [5] Cumulative-PPL exponential fit  y(x) = A * r^x + C
#     cumulative PPLs (1..n) = [117.856, 30.262, 17.785, 18.457, 20.971, 15.102]
#     A = 678.2175 | r = 0.151464 | C = 15.0921
#     fit quality: R^2 = 0.9946 | RMSE = 2.6845

# [6] Context Progression Rates  CPR_x = r / x^2  (r from [5], NOT CSR)
#     DCPR  = r / Dist-2^2 = 0.1515 / (0.9224)^2 = 0.1515 / 0.8508 = 0.1780
#     TTCPR = r / TTR^2 = 0.1515 / (0.6435)^2 = 0.1515 / 0.4141 = 0.3658
#     VCPR  = r / VR^2 = 0.1515 / (0.1043)^2 = 0.1515 / 0.0109 = 13.9105
#     CCPR  = r / CD^2 = 0.1515 / (0.0957)^2 = 0.1515 / 0.0091 = 16.5547
#     CECPR = r / CE^2 = 0.1515 / (0.0261)^2 = 0.1515 / 0.0007 = 222.5684

# ======================================================================
# DETAILED WALKTHROUGH: Strange
# ======================================================================
# Sentences detected: 6
# Shuffled (seed=42): Feature X is important, feature Y is less important. Therefore, the outcome is what it is, because the model made a prediction. As a result,...

# [1] Perplexity (PPL = exp(cross-entropy loss))
#     PPL_original  = 17.3709
#     PPL_shuffled  = 16.4372

# [2] Continuous Structure Rate  CSR = (PPL_shuffled - PPL_original) / PPL_original
#     CSR = (16.4372 - 17.3709) / 17.3709 = -0.9337 / 17.3709 = -0.053753

# [3] Diversity / structural measures
#     Dist-2 (distinct bigram ratio) = 0.9365
#     TTR    (type-token ratio)      = 0.5781
#     VR     (verb ratio)            = 0.1875
#     CD     (connectives density)   = 0.1250
#     CE     (cause-effect ratio)    = 0.1094

# [4] Fluency-Diversity Rate  FDR = (Dist-2)^2 / ln(PPL_original)
#     FDR = (0.9365)^2 / ln(17.3709) = 0.8770 / 2.8548 = 0.3072

# [5] Cumulative-PPL exponential fit  y(x) = A * r^x + C
#     cumulative PPLs (1..n) = [53.555, 20.167, 21.204, 18.606, 18.016, 17.371]
#     A = 370.7980 | r = 0.097514 | C = 17.3609
#     fit quality: R^2 = 0.9854 | RMSE = 1.5631

# [6] Context Progression Rates  CPR_x = r / x^2  (r from [5], NOT CSR)
#     DCPR  = r / Dist-2^2 = 0.0975 / (0.9365)^2 = 0.0975 / 0.8770 = 0.1112
#     TTCPR = r / TTR^2 = 0.0975 / (0.5781)^2 = 0.0975 / 0.3342 = 0.2918
#     VCPR  = r / VR^2 = 0.0975 / (0.1875)^2 = 0.0975 / 0.0352 = 2.7737
#     CCPR  = r / CD^2 = 0.0975 / (0.1250)^2 = 0.0975 / 0.0156 = 6.2409
#     CECPR = r / CE^2 = 0.0975 / (0.1094)^2 = 0.0975 / 0.0120 = 8.1514