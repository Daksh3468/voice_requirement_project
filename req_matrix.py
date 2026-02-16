import re
import pandas as pd

# -----------------------------------
# CONFIGURATION
# -----------------------------------

AMBIGUOUS_WORDS = [
    "fast", "quick", "efficient", "user-friendly",
    "etc", "and so on", "appropriate", "suitable",
    "robust", "secure", "easy", "optimize",
    "low latency", "large number", "reliable",
    "consistent", "scalable"
]

ACTION_VERBS = [
    "shall", "must", "should", "will", "allow",
    "provide", "generate", "store", "process",
    "validate", "send", "receive", "display",
    "ensure", "support"
]

# -----------------------------------
# PARSER
# -----------------------------------

def parse_requirements(raw_text):
    functional = []
    non_functional = []

    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("[Functional]"):
            functional.append(line.replace("[Functional]", "").strip())
        elif line.startswith("[Non-Functional]"):
            non_functional.append(line.replace("[Non-Functional]", "").strip())

    return functional, non_functional


# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

def contains_action(req):
    return any(verb in req.lower() for verb in ACTION_VERBS)

def contains_ambiguity(req):
    return any(word in req.lower() for word in AMBIGUOUS_WORDS)

def contains_measurable_value(req):
    return bool(re.search(r"\d+|\%|seconds|ms|users|transactions|requests", req.lower()))


# -----------------------------------
# SCORING LOGIC
# -----------------------------------

def evaluate_completeness(requirements, is_nfr=False):
    missing = 0

    for req in requirements:
        if not contains_action(req):
            missing += 1
        if is_nfr and not contains_measurable_value(req):
            missing += 1

    if missing > 3:
        return "Poor"
    elif 2 <= missing <= 3:
        return "Fair"
    elif missing == 1:
        return "Sufficient"
    elif missing == 0:
        return "Good"
    else:
        return "Excellent"


def evaluate_correctness(requirements):
    if len(requirements) == 0:
        return "Poor"

    incorrect = sum(1 for r in requirements if contains_ambiguity(r))
    percentage = (incorrect / len(requirements)) * 100

    if percentage > 50:
        return "Poor"
    elif 30 <= percentage <= 50:
        return "Fair"
    elif 10 <= percentage < 30:
        return "Sufficient"
    elif percentage < 10 and percentage > 0:
        return "Good"
    else:
        return "Excellent"


def evaluate_clarity(requirements):
    unclear = 0

    for req in requirements:
        if len(req.split()) < 6 or contains_ambiguity(req):
            unclear += 1

    if unclear > 3:
        return "Poor"
    elif 2 <= unclear <= 3:
        return "Fair"
    elif unclear == 1:
        return "Sufficient"
    elif unclear == 0:
        return "Good"
    else:
        return "Excellent"


# -----------------------------------
# MATRIX GENERATOR
# -----------------------------------

def generate_validation_matrix(raw_text):
    functional, non_functional = parse_requirements(raw_text)

    data = {
        "Criteria": ["Completeness", "Correctness", "Clarity"],
        "Functional Requirements": [
            evaluate_completeness(functional),
            evaluate_correctness(functional),
            evaluate_clarity(functional)
        ],
        "Non-Functional Requirements": [
            evaluate_completeness(non_functional, is_nfr=True),
            evaluate_correctness(non_functional),
            evaluate_clarity(non_functional)
        ]
    }

    return pd.DataFrame(data)


# -----------------------------------
# EXAMPLE USAGE
# -----------------------------------

if __name__ == "__main__":

    input_text = """
  
   
    [Functional] The system shall allow users to create a personal profile with name, address, and health-related information
    [Functional] The system shall provide a user-friendly interface with good UI for users to input their fitness goals and track progress
    [Non-Functional] The system shall ensure user data is stored securely and in compliance with relevant regulations such as GDPR and HIPAA
    [Functional] The system shall provide personalized fitness recommendations based on user input and progress using any suitable algorithms or methodologies
    [Non-Functional] The system shall be compatible with both iOS and Android mobile platforms
    [Functional] The system shall track user progress using daily improvements and other relevant metrics
    [Functional] The system shall apply validation rules and checks to user input for fitness goals and progress tracking
    [Functional] The system shall implement error handling mechanisms to handle user errors or inconsistencies in input by re-measuring the stats
    [Non-Functional] The system shall ensure the accuracy and integrity of user data through suitable mechanisms
    [Functional] The system shall be a sensor-based application

    """

    matrix = generate_validation_matrix(input_text)

    print("\nREQUIREMENTS VALIDATION MATRIX\n")
    print(matrix.to_string(index=False))
