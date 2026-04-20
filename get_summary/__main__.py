"""
Main script to present the results.
"""

import subprocess


def main():
    # summary table for NLP metrics
    subprocess.run(["python", "-m", "get_summary.summary_table"])

    # summary table and charts for perplexity
    subprocess.run(["python", "-m", "get_summary.summary_ppl"])


if __name__ == "__main__":
    main()
