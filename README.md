# **On the Importance and Evaluation of Narrativity in Natural Language AI Explanations**
M. Cedro, D. Martens (2026) - University of Antwerp, Belgium.

## Abstract

Explainable AI (XAI) aims to make the behaviour of machine learning models interpretable, yet many explanation methods remain difficult to understand. The integration of Natural Language Generation into XAI aims to deliver explanations in textual form, making them more accessible to practitioners. Current approaches, however, largely yield static lists of feature importances. Although such explanations indicate *what* influences the prediction, they do not explain *why* the prediction occurs. In this study, we draw on insights from social sciences and linguistics, and argue that XAI explanations should be presented in the form of narratives. Narrative explanations support human understanding through four defining properties: **continuous structure**, **cause–effect mechanisms**, **linguistic fluency**, and **lexical diversity**. We show that standard NLP metrics based solely on token probability or word frequency fail to capture these properties and can be matched or exceeded by tautological text that conveys no explanatory content. To address this, we propose **seven automatic metrics** that quantify the narrative quality of explanations along the four identified dimensions. We benchmark current state-of-the-art explanation generation methods on six datasets and show that the proposed metrics separate descriptive from narrative explanations more reliably than standard NLP metrics. Finally, we propose a set of problem-agnostic **XAI Narrative generation rules** for producing natural language XAI explanations, so that the resulting XAI Narratives exhibit stronger narrative properties and align with findings from the linguistic and social science literature.

## Proposed narrative-quality metrics

The seven metrics proposed in the paper, grouped by the narrative property they capture:

| Property              | Metric  | Definition                                                     | Direction |
|-----------------------|---------|----------------------------------------------------------------|-----------|
| Linguistic fluency    | **FDR**   | Fluency–Diversity Rate: `(Dist-2)^2 / ln(PPL)`                 | ↑         |
| Continuous structure  | **CSR**   | Continuous Structure Rate: `(PPL_shuffled − PPL) / PPL`        | ↑         |
| Continuous structure  | **DCPR**  | `r / Dist-2^2` - Diversity-adjusted Context Progression Rate   | ↓         |
| Cause–effect          | **CECPR** | `r / CE^2` - Cause-Effect-adjusted CPR                         | ↓         |
| Cause–effect          | **CCPR**  | `r / CD^2` - Connectives-adjusted CPR                          | ↓         |
| Lexical diversity     | **TTCPR** | `r / TTR^2` - Type-Token-adjusted CPR                          | ↓         |
| Lexical diversity     | **VCPR**  | `r / VR^2` - Verb-adjusted CPR                                 | ↓         |

Here `r` is the geometric decay rate from fitting `y(x) = A · r^x + C` to cumulative sentence-level perplexities, `PPL_shuffled` is the perplexity under a seeded sentence shuffle, `Dist-2` is the distinct-bigram ratio, `TTR` the type–token ratio, `VR` the verb ratio, `CD` the connectives density, and `CE` the cause–effect marker ratio.

## Repository structure

```
.
├── config.yaml                  # Global config (datasets, model hyper-params, LLMs, paths)
├── data/
│   ├── datasets/                # 6 tabular datasets (CSV)
│   ├── SHAP_results/            # Per-dataset SHAP feature attributions
│   ├── llm_inputs/              # Prompt inputs for each generator
│   ├── llm_outputs/             # Raw LLM-generated explanations
│   └── template_outputs/        # Outputs for templated baselines
├── get_feature_attributions/    # Train RF models and compute SHAP values
├── get_narratives/              # Build prompts, call LLMs, generate templated text
├── get_metrics/                 # Seven narrative metrics + standard NLP baselines
├── get_summary/                 # Summary tables, critical-difference diagrams, charts
├── utils/                       # Lexicons (connectives, cause-effect), PPL utilities
├── summary/                     # Aggregated summaries + plotting scripts
├── results/                     # Per-method, per-dataset metric results
└── tutorial/
    └── demo.py                  # Stand-alone demo: score any text across all metrics
```

## Datasets

Six binary classification datasets, all included under [data/datasets](data/datasets):

| Dataset         | Rows  | Cols | Target                 | Test AUC |
|-----------------|-------|------|------------------------|----------|
| compas          | 6172  | 10   | recidivism             | 0.850    |
| diabetes        | 768   | 9    | diabetes diagnosis     | 0.829    |
| fifa            | 128   | 24   | Man of the Match       | 0.776    |
| german_credit   | 1000  | 11   | credit risk            | 0.671    |
| student         | 395   | 33   | pass/fail              | 0.958    |
| stroke          | 5110  | 11   | stroke                 | 0.848    |

## Benchmarked explanation methods

- **Explingo** - LLM rewriting of SHAP feature lists.
- **Explingo (zero-shot)** - no in-context examples.
- **XAIstories** - narrative SHAP explanations via LLM.
- **TalkToModel** - dynamic templated prompts (see [TalkToModel repo](https://github.com/dylan-slack/TalkToModel)).
- **Templated Narrative** - deterministic template baseline.
- **XAINarratives** - narrative explanations produced with the **XAI Narrative generation rules** proposed in the paper.

## Installation

```bash
git clone https://github.com/ADMAntwerp/On-the-Importance-and-Evaluation-of-Narrativity-in-Natural-Language-AI-Explanations.git
cd On-the-Importance-and-Evaluation-of-Narrativity-in-Natural-Language-AI-Explanations
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # see note below
```

The code depends on `numpy`, `pandas`, `scikit-learn`, `shap`, `scipy`, `torch`, `transformers`, `nltk`, `spacy`, `pyyaml`, and `openai`. A pinned `requirements.txt` will be added in the release.

Set your API key for the LLM-based generators (only needed to re-run generation, not evaluation):

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

## Reproducing the paper

Each stage is an importable Python package runnable with `python -m`.

```bash
# 1. Train RF models and compute SHAP feature attributions
python -m get_feature_attributions

# 2. Generate natural-language explanations (LLM + template baselines)
python -m get_narratives

# 3. Compute all seven narrative metrics + standard NLP baselines
python -m get_metrics

# 4. Build summary tables and plots
python -m get_summary
```

Key knobs live in [config.yaml](config.yaml): the generation LLM (`MODEL`, default `gpt-4.1`), the perplexity model (`MODEL_PPL`, default `meta-llama/Llama-3.1-8B`), the number of top SHAP features (`TOP_FEATURES`), and the per-dataset test-instance cap (`INSTANCE_LIMIT`).

## Quick demo: score any text

[tutorial/demo.py](tutorial/demo.py) loads a local LLM for perplexity and prints all seven metrics plus standard NLP baselines for a description-style vs narrative-style vs tautological XAI text:

```bash
python -m tutorial.demo
```

The script also prints a per-metric calculation walkthrough (intermediate PPLs, the fitted `A, r, C`, and the ratio substitutions) which is useful for understanding or debugging the metrics on your own texts.

## Citation

```bibtex
@article{cedro2026importance,
  title={On the Importance and Evaluation of Narrativity in Natural Language AI Explanations},
  author={Cedro, Mateusz and Martens, David},
  journal={arXiv preprint arXiv:2604.18311},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).
