"""
Main script to transform SHAP to the natural language explanations.
"""

import subprocess


def main():
    # # run utils.postprocess_test_samples.py
    # subprocess.run(["python", "-m", "get_narratives.postprocess_test_samples"])

    # # prepare LLM inputs
    # subprocess.run(["python", "-m", "get_narratives.prepare_llm_inputs"])

    # call LLM to get narratives
    # subprocess.run(["python", "-m", "get_narratives.llm_call"]) # commented out to avoid unnecessary API calls

    # # # prepare templates
    subprocess.run(["python", "-m", "get_narratives.prepare_templates"])


if __name__ == "__main__":
    main()
