import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import yaml
from utils.prompt_utils import prompt_configs, narrative_rules, xaistories_rules

load_dotenv()

with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    TOP_FEATURES = int(config_data["TOP_FEATURES"])
    N_SENTENCES = int(config_data["N_SENTENCES"])
    INSTANCE_LIMIT = int(config_data.get("INSTANCE_LIMIT", 1))
    LLM_INPUTS_FOLDER = config_data["LLM_INPUTS_FOLDER"]
    LLM_OUTPUTS_FOLDER = config_data["LLM_OUTPUTS_FOLDER"]
    MODEL = config_data["MODEL"]
    TEMPERATURE = config_data["TEMPERATURE"]
    NUCLEUS_TOP_P = config_data["NUCLEUS_TOP_P"]

api_key = os.getenv("OPENAI_API_KEY")


def main():
    dataset_names = list(config_data["DATASETS_INFO"].keys())
    methods = [
        #"xaistories",
        # "xaistories_narratives",
        # "explingo",
        # "explingo_zero_shot",
        "explingo_narratives",
    ]

    # # Check if LLM_OUTPUTS_FOLDER/{method} exists and is not empty, so we don't overwrite previous results
    # for method in methods:
    #     method_folder = os.path.join(LLM_OUTPUTS_FOLDER, method)
    #     if os.path.exists(method_folder) and os.listdir(method_folder):
    #         raise ValueError(f"LLM_OUTPUTS_FOLDER/{method} is not empty: {method_folder}")

    for method in methods:
        if method == "explingo":
            from explingo import Narrator

            for ds_name in dataset_names:
                # read the example narratives from the LLM_INPUTS_FOLDER with method and ds_name
                print(f"Processing {ds_name} with {method}...")
                llm_inputs = os.path.join(
                    LLM_INPUTS_FOLDER, f"{method}/{ds_name}_llm_inputs.json"
                )
                with open(llm_inputs, "r") as f:
                    explingo_inputs = json.load(f)

                results = {}
                for i, explingo_input in enumerate(explingo_inputs):
                    example_narratives = explingo_input["example_narratives"]
                    explanation_format = explingo_input["explanation_format"]
                    context = explingo_input["context"]
                    explanation = explingo_input["explanation"]

                    narrator = Narrator(
                        openai_api_key=api_key,
                        explanation_format=explanation_format,
                        context=context,
                        sample_narratives=example_narratives,
                        gpt_model_name=MODEL,
                        temperature=TEMPERATURE,
                        top_p=NUCLEUS_TOP_P,  # Nucleus sampling parameter
                    )

                    narrative = narrator.narrate(explanation)
                    results[i] = narrative
                    # print(narrative)

                    time.sleep(1)  # to avoid rate limits
                    if i + 1 >= INSTANCE_LIMIT:
                        break

                output_folder = os.path.join(LLM_OUTPUTS_FOLDER, method)

                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"{ds_name}.json")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

        if method == "explingo_narratives": # "explingo_zero_shot"
            from explingo import Narrator

            for ds_name in dataset_names:
                # read the example narratives from the LLM_INPUTS_FOLDER with method and ds_name
                print(f"Processing {ds_name} with {method}...")
                llm_inputs = os.path.join(
                    LLM_INPUTS_FOLDER,
                    f"explingo/{ds_name}_llm_inputs.json",  # use explingo inputs but without example narratives
                )
                with open(llm_inputs, "r") as f:
                    explingo_inputs = json.load(f)

                results = {}
                for i, explingo_input in enumerate(explingo_inputs):
                    # example_narratives = explingo_input["example_narratives"]
                    explanation_format = explingo_input["explanation_format"]
                    context = explingo_input["context"]
                    explanation = explingo_input["explanation"]

                    context_with_rules = f"{context}\n\n{narrative_rules}"

                    narrator = Narrator(
                        openai_api_key=api_key,
                        explanation_format=explanation_format,
                        context=context_with_rules,
                        # sample_narratives=example_narratives,
                        gpt_model_name=MODEL,
                        temperature=TEMPERATURE,
                        top_p=NUCLEUS_TOP_P,  # Nucleus sampling parameter
                    )

                    try:
                        narrative = narrator.narrate(explanation)
                    except AttributeError:
                        # If parsing fails, get the raw LLM output
                        full_prompt = narrator._assemble_prompt(
                            narrator.default_prompt, explanation, explanation_format, examples=None
                        )
                        narrative = narrator.llm(full_prompt)[0]
                    results[i] = narrative
                    # print(narrative)
                    time.sleep(1)  # to avoid rate limits
                    if i + 1 >= INSTANCE_LIMIT:
                        break

                output_folder = os.path.join(LLM_OUTPUTS_FOLDER, method)

                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"{ds_name}.json")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

        if method == "xaistories_narratives": # "xaistories"
            client = OpenAI(api_key=api_key)

            for ds_name in dataset_names:
                print(f"Processing {ds_name} with {method}...")
                llm_inputs = os.path.join(
                    LLM_INPUTS_FOLDER, f"xaistories/{ds_name}_llm_inputs.json"
                )
                with open(llm_inputs, "r") as f:
                    xaistories_inputs = json.load(f)

                all_results = {}
                for i, xaistories_input in enumerate(xaistories_inputs):
                    prompt = (
                        f"Task: {xaistories_input['task_description']}\n"
                        f"Predicted Class: {xaistories_input['predicted_class']}\n"
                        f"Dataset Description: {xaistories_input['dataset_description']}\n"
                        f"Target Description: {xaistories_input['target_description']}\n"
                        "Feature Details:\n"
                    )
                    # Include only the top TOP_FEATURES features
                    for feat in xaistories_input["feature_desc"][:TOP_FEATURES]:
                        prompt += (
                            f"- {feat['feature_name']}: Importance: {feat['feature_importance']}, "
                            f"Value: {feat['feature_value']}, Average: {feat['feature_average']}, "
                            f"Description: {feat['feature_description']}\n"
                        )

                    rules_formatted = xaistories_rules.format(
                        sentence_limit=N_SENTENCES,
                        num_feat=TOP_FEATURES,
                        target_instance=prompt_configs[ds_name]["target_description"],
                    )
                    prompt += (
                        "\nPlease generate an explanation that follows these rules:\n"
                        + rules_formatted
                        + "\n\n"
                        + narrative_rules
                    )

                    response = client.chat.completions.create(
                        model=MODEL, 
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert in explaining model predictions in a form of a narrative.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=TEMPERATURE,
                        top_p=NUCLEUS_TOP_P,  # Nucleus sampling
                        seed=42,
                        max_tokens=2000,
                        n=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    llm_output = response.choices[0].message.content.strip()
                    all_results[i] = llm_output
                    # print(llm_output)
                    time.sleep(1)
                    if i + 1 >= INSTANCE_LIMIT:
                        break
                # Save the results
                output_folder = os.path.join(LLM_OUTPUTS_FOLDER, method)
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"{ds_name}.json")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
