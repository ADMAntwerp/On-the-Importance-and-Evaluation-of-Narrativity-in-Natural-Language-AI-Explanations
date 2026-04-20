"""
task_description: A high-level description of why we're explaining the prediction.
dataset_description: Describes the dataset, the prediction target, and the overall prediction task.
target_description: Description of the target instance.
feature_desc: List of feature names and descriptions (as a list of dictionaries). This can be passed to the LLM when needed, ensuring it understands what each feature means.
"""

prompt_configs = {
    # COMPAS configuration
    "compas": {
        "task_description": (
            "The prediction task is to estimate the risk of recidivism (whether an individual will be rearrested within two years). "
        ),
        "dataset_description": (
            "The dataset contains data on individuals processed in the US criminal justice system. "
            "Features include demographic information, prior criminal history, and details related to prior charges."
        ),
        "target_description": ("An individual undergoing recidivism risk assessment."),
        "feature_desc": [
            {"age": "Age of the individual"},
            {"number_of_prior_crimes": "Number of prior crimes committed"},
            {"months_in_jail": "Number of months spent in jail"},
            {"felony": "Indicator if the current charge is a felony"},
            {"misdemeanor": "Indicator if the current charge is a misdemeanor"},
            {"woman": "Indicator if the individual is a woman"},
            {"black": "Indicator if the individual is recorded as Black"},
            {"recidivated": "Indicator if the individual has recidivated before"},
        ],
    },
    # Diabetes configuration
    "diabetes": {
        "task_description": "The prediction task is to estimate whether the patient has diabetes.",
        "dataset_description": (
            "This dataset contains health records of patients, including clinical measures and demographic information. "
        ),
        "target_description": (
            "A patient whose clinical and demographic data are used to predict diabetes risk"
        ),
        "feature_desc": [
            {"Pregnancies": "Number of pregnancies"},
            {"Glucose": "Plasma glucose concentration"},
            {"BloodPressure": "Diastolic blood pressure (mm Hg)"},
            {"SkinThickness": "Triceps skin fold thickness (mm)"},
            {"Insulin": "2-Hour serum insulin (mu U/ml)"},
            {"BMI": "Body Mass Index (kg/m^2)"},
            {"DiabetesPedigreeFunction": "Diabetes pedigree function"},
            {"Age": "Age of the patient"},
        ],
    },
    # FIFA configuration
    "fifa": {
        "task_description": (
            "The task was to predict whether a football team will receive or not receive the 'Man of the Match' award for its player."
        ),
        "dataset_description": (
            "This dataset contains match performance data from FIFA World Cup matches. "
        ),
        "target_description": ("A team in a football match."),
        "feature_desc": [
            {"Team": "Name of the team"},
            {"Opponent": "Name of the opponent team"},
            {"Goal Scored": "Number of goals scored"},
            {"Ball Possession %": "Percentage of ball possession"},
            {"Attempts": "Number of attempts"},
            {"On-Target": "Number of on-target attempts"},
            {"Off-Target": "Number of off-target attempts"},
            {"Blocked": "Number of blocked attempts"},
            {"Corners": "Number of corner kicks"},
            {"Offsides": "Number of offsides"},
            {"Free Kicks": "Number of free kicks"},
            {"Saves": "Number of saves made"},
            {"Pass Accuracy %": "Pass accuracy percentage"},
            {"Passes": "Total number of passes"},
            {
                "Distance Covered (Kms)": "Total distance covered by the team in kilometers"
            },
            {"Fouls Committed": "Number of fouls committed"},
            {"Yellow Card": "Number of yellow cards received"},
            {"Yellow & Red": "Indicator for yellow and red card incidents"},
            {"Red": "Indicator for red card incidents"},
            {"1st Goal": "Indicator if the player/team scored the first goal"},
            {"Round": "Stage of the tournament (Group, Knockout, etc.)"},
            {"PSO": "Indicator related to penalty shoot-out"},
            {"Goals in PSO": "Number of goals scored in the penalty shoot-out"},
            {"Own goals": "Number of own goals"},
            {"Own goal Time": "Time at which an own goal occurred"},
        ],
    },
    # German Credit configuration
    "german_credit": {
        "task_description": (
            "The prediction task is to classify applicants as 'good' or 'bad' credit risks."
        ),
        "dataset_description": (
            "This dataset contains information on individuals applying for credit, including financial status, credit history, and personal details. "
        ),
        "target_description": (
            "A credit applicant whose financial and personal details are used to assess credit risk."
        ),
        "feature_desc": [
            {"Age": "Age of the applicant"},
            {"Sex": "Gender of the applicant"},
            {
                "Job": "Job type or category"
            },  # numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled
            {"Housing": "Housing status (Own, Rent, etc.)"},
            {"Saving accounts": "Status of the applicant's saving accounts"},
            {"Checking account": "Status of the applicant's checking account"},
            {"Credit amount": "Amount of credit requested"},
            {"Duration": "Duration of the loan in months"},
            {"Purpose": "Purpose of the loan"},
        ],
    },
    # Student configuration
    "student": {
        "task_description": (
            "The task is to predict whether the student will pass or fail the mathematics course. "
        ),
        "dataset_description": (
            "This dataset contains student information, including demographic background, school performance, and family context. "
        ),
        "target_description": ("A student undertaking the mathematics course."),
        "feature_desc": [
            {
                "school": "School attended"
            },  #  school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveir
            {"sex": "Gender of the student"},
            {"age": "Age of the student"},
            {
                "address": "Type of address (urban/rural)"
            },  # (binary: 'U' - urban or 'R' - rural)
            {
                "famsize": "Family size"
            },  # (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
            {
                "Medu": "Mother's education level"
            },  # numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
            {
                "Fedu": "Father's education level"
            },  #  (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
            {"Mjob": "Mother's job"},
            {"Fjob": "Father's job"},
            {"reason": "Reason for choosing the school"},
            {"guardian": "Guardian of the student"},
            {
                "traveltime": "Travel time from home to school"
            },  # numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
            {
                "studytime": "Weekly study time"
            },  # (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
            {"failures": "Number of past class failures"},
            {"schoolsup": "Extra educational support provided by school"},
            {"famsup": "Family educational support"},
            {"paid": "Extra paid classes"},
            {"activities": "Participation in extracurricular activities"},
            {"nursery": "Attendance at nursery school"},
            {"higher": "Desire to pursue higher education"},
            {"internet": "Internet access at home"},
            {"romantic": "Relationship status (in a romantic relationship or not)"},
            {
                "famrel": "Quality of family relationships"
            },  # (numeric: from 1 - very bad to 5 - excellent)
            {
                "freetime": "Amount of free time after school"
            },  # (numeric: from 1 - very low to 5 - very high)
            {
                "goout": "Frequency of going out with friends"
            },  # (numeric: from 1 - very low to 5 - very high)
            {
                "Dalc": "Workday alcohol consumption"
            },  # numeric: from 1 - very low to 5 - very high)
            {
                "Walc": "Weekend alcohol consumption"
            },  # (numeric: from 1 - very low to 5 - very high)
            {
                "health": "Current health status"
            },  # (numeric: from 1 - very bad to 5 - very good)
            {"absences": "Number of school absences"},
            {"G1": "First period grade"},
            {"G2": "Second period grade"},
            {"G3": "Final grade"},
            {
                "Pstatus": "Parent's cohabitation status"
            },  # (binary: 'T' - living together or 'A' - apart)
        ],
    },
    # Stroke configuration
    "stroke": {
        "task_description": (
            "The taks was to predict a risk of stroke for individuals based on their health records."
        ),
        "dataset_description": (
            "This dataset contains patient demographic and health information, including lifestyle and medical history factors. "
        ),
        "target_description": (
            "A patient whose health and lifestyle factors are used to predict stroke risk."
        ),
        "feature_desc": [
            {"gender": "Gender of the patient"},
            {"age": "Age of the patient"},
            {
                "hypertension": "Indicator if the patient has hypertension"  # (0 = No, 1 = Yes)
            },
            {
                "heart_disease": "Indicator if the patient has heart disease"  # (0 = No, 1 = Yes)
            },
            {"ever_married": "Marital status"},
            {"work_type": "Type of work or employment status"},
            {"Residence_type": "Type of residence (Urban/Rural)"},
            {"avg_glucose_level": "Average glucose level"},
            {"bmi": "Body Mass Index"},
            {"smoking_status": "Smoking status category"},
        ],
    },
}


list_binary = [
    "felony",
    "misdemeanor",
    "woman",
    "black",
    "recidivated",
    "Yellow & Red",
    "Red",
    "1st Goal",
    "PSO",
    "hypertension",
    "heart_disease",
]


narrative_rules = """
Generate a narrative explanation (an XAI Narrative) based on the following rules:
1. An XAI Narrative should establish a continuous structure by following a clear narrative arc with a beginning, middle, and end, while using explicit linguistic connectives so that individual events can be seen in the perspective of the others.
2. An XAI Narrative should explicitly identify the underlying cause-effect mechanisms to clarify why the system made a particular prediction.
3. An XAI Narrative should explain model's prediction with linguistic fluency, avoiding repetitive, list-like structures.
4. An XAI Narrative should use a lexically diverse vocabulary with an emphasis on active verbs to express how specific features influence the final prediction.
"""



xaistories_rules = """
**Format-related rules**:
1) Start the explanation immediately.
2) Limit the entire answer to exactly {sentence_limit} sentences.
3) Only mention the top {num_feat} most important features in the narrative.
4) Do not use tables or lists, or simply rattle through the features and/or nodes one by one. The goal is to have a narrative/story.

**Content related rules**:
1) Be clear about what the model actually predicted for the {target_instance}.
2) Discuss how the features contributed to final prediction. Make sure to clearly establish this the first time you refer to a feature. 
3) Consider the feature importance, feature values, and averages when referencing their relative importance.
4) Begin the discussion of features by presenting those with the highest absolute feature importance values first. The reader should be able to tell what the order of importance of the features is based on their feature importance value.
5) Provide a suggestion or interpretation as to why a feature contributed in a certain direction. Try to introduce external knowledge that you might have.
6) If there is no simple explanation for the effect of a feature, consider the context of other features in the interpretation.
7) Do not use the feature importance numeric values in your answer.
8) You can use the feature values themselves in the explanation, as long as they are not categorical variables. If they are an categorical variable, refer to the semantic meaning of the category from the feature_desc.
9) Do not refer to the average for every single feature; reserve it for features where it truly clarifies the explanation.
10) When you refer to an instance, keep in mind that the target instance is a {target_instance}.
11) Tell a clear and engaging story, including details from both feature values and node connections, to make the explanation more relatable and interesting.
12) Use clear and simple language that a general audience can understand, avoiding overly technical jargon or explaining any necessary technical terms in plain language.
"""


# xainarratives_rules = """
# **Format-related rules**:
# 1) Start the explanation immediately with a clear statement of the prediction outcome.
# 2) Limit the entire answer to exactly {sentence_limit} sentences.
# 3) Only mention the top {num_feat} most important features in the narrative.
# 4) Do not use tables, bullet points, or lists. The output must be a fluid paragraph.

# **Narrative Structure & Story-telling**:
# 1) **Opening Sentence - Prediction Only**: The first sentence must introduce only the prediction outcome for this {target_instance}. Do not mention any features or reasons in this opening sentence. Simply state what the model predicted.
# 2) **Tell the Story Leading to the Prediction**: After the opening, structure the explanation as a narrative that traces the sequence of events/conditions leading to the model's decision, not just a listing of features. Show how one condition leads to or interacts with another.
# 3) **Clear Narrative Arc**: 
#    - Beginning: First sentence states the prediction outcome only.
#    - Middle: Explain the primary drivers, supporting factors, interactions, or mitigating circumstances that add nuance.
#    - End: Close with a synthesis that ties together how all discussed factors collectively led to the final prediction.
# 4) **Knowledge Why, Not Just Knowledge That**: Go beyond stating what features are important. Explain the underlying mechanisms and reasons why these feature values causally contribute to the outcome in this specific case.
# 5) **Avoid Technical Description**: This is an explanation, not a description. Do not simply enumerate feature values like a manual. Instead, weave them into a coherent story about why this particular instance received this prediction.

# **Connectives & Causal Structure**:
# 1) **Use Connectives Explicitly**: Use contingency connectives (e.g., 'because', 'thus', 'therefore', 'as a result', 'consequently', 'leading to') to show causality, and expansion connectives (e.g., 'moreover', 'additionally', 'specifically', 'furthermore') to add detail. Do not leave sentences as isolated facts.
# 2) **Make Causal Links Explicit**: Use discourse markers to show how features causally relate to each other and to the outcome (e.g., "because of this," "which in turn," "consequently," "as a result of").
# 3) **Temporal/Logical Progression**: Arrange your discussion so that there is a logical flow from conditions → intermediate effects → final outcome. The reader should be able to follow the chain of reasoning.
# 4) **Show Interactions and Trade-offs**: When features have opposing influences, explicitly discuss how they interact or counterbalance each other (e.g., "although X is favorable, it cannot compensate for Y").

# **Continuity & Coherence**:
# 1) **Ensure Flow**: Ensure the text flows as a single coherent story with a clear beginning, middle, and end. Avoid "jumping" between unrelated features without a transitional phrase.
# 2) **No Isolated Facts**: Every sentence should connect to what came before and what comes after. Avoid abrupt topic shifts without transitional phrases.
# 3) **Interweave Threads**: When discussing multiple features, interweave their discussion to show relationships rather than treating each as a separate, independent section.
# 4) **Maintain Focus**: Keep the explanation focused on explaining this specific prediction. Avoid generic statements that could apply to any instance.
# 5) **Unified Subject**: Maintain focus on the specific instance throughout. The narrative should feel like the story of this particular case, not a general discussion of the model.

# **Lexical Diversity & Dynamic Language**:
# 1) **Verbs over Nouns**: Use active verbs to describe how features *act* upon the prediction (e.g., "limits," "boosts," "offsets," "drives," "undermines," "amplifies," "constrains") rather than static descriptions (e.g., "is high," "is important").
# 2) **Vary Your Vocabulary**: Avoid repetitive phrases. Use synonyms and varied sentence structures to maintain reader engagement. Do not recycle the same feature names verbatim multiple times.
# 3) **Prioritize Dynamic Verbs**: Emphasize verbs that convey action and change rather than static state descriptions. This helps convey how features actively shape the prediction.
# 4) **Avoid Template-Like Language**: Each explanation should feel unique to the instance, not generated from a rigid template. Adapt your phrasing to the specific case.
# 5) **Conversational Tone**: Use person-oriented, familiar language rather than technical jargon.

# **Content & Explanation Quality**:
# 1) **Explain Through Mechanisms**: For each feature discussed, identify and explain the real-world causal mechanism that connects this feature value to the prediction outcome. Show the chain: feature value → mechanism → intermediate effect → final prediction. Do not just state correlation or importance.
# 2) **Contextualize with Real-World Scenarios**: Help the reader build a mental model by relating feature values to real-world scenarios or typical cases (e.g., "unlike most applicants who...", "in typical circumstances...").
# 3) **Provide Necessary Background**: When needed, briefly elaborate on what a feature represents or why it matters in this domain before discussing its contribution. This helps integrate new information into familiar patterns.
# 4) **Importance Rank and Logical Flow**: Begin the discussion of features by presenting those with the highest absolute feature importance values first. The reader should be able to tell what the order of importance of the features is based on their feature importance value.
# 5) **Synthesize and Show Interplay**: Features rarely act in isolation. Explicitly discuss how features interact, reinforce, or counteract each other. Show how the combination of feature values creates the specific prediction through their joint mechanisms, not just their individual effects.
# 6) **Constraint**: Do not use the feature importance numeric values in your answer.
# 7) **Constraint**: You can use the feature values themselves, but if they are categorical, refer to their semantic meaning rather than the encoded value.

# **What to Avoid**:
# 1) **No Feature Lists**: Never present features as "Feature X is important, Feature Y is important, Feature Z is important." This is description, not explanation.
# 2) **No Ranking Statements**: Avoid phrases like "the most important feature," "the second most important feature" unless embedded naturally in causal discussion.
# 3) **No Mechanical Language**: Avoid phrases that sound like model output (e.g., "has a positive influence on the prediction," "increases the predicted value"). Instead, explain the real-world mechanism.
# 4) **No Repetitive Structure**: Each sentence should advance the narrative, not repeat the same pattern with different feature names.
# 5) **No Generic Statements**: Avoid statements that could apply to any instance. Make the explanation specific to this case's unique combination of feature values.
# 6) **No Features in Opening Sentence**: The first sentence must only state the prediction. Do not combine the prediction with feature explanations or reasons in the opening.
# """
