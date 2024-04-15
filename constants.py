if __name__ == "__main__":
    print("This file is used for data storing purpose only.")
    exit(0)


# mimic PATH (in which there are hosp and icu data)
from pathlib import Path
from pandasql import PandaSQL
from secret import MIMIC_PATH_STR, POSTGRESQL_CONNECTION_STRING


MIMIC_PATH = Path(MIMIC_PATH_STR)

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

IMPORTANT_MESUREMENTS_ICU = {
    227519: "urine_output",
    224639: "weight",
    227457: "plt",
    220615: "creatinine",
}

PREPROCESSED_MESUREMENTS = ["sugar"]

IMPORTANT_MESUREMENTS_LABEVENT = {51006: "bun"}

# Define ICD-9/10 codes for DKA
DKA_CODE_V9 = [
    "24910",  # Secondary diabetes mellitus with ketoacidosis
    "24911",
    "25010",
    "25011",
    "25012",
    "25013",
]
DKA_CODE_V10 = [
    "E081",  # Diabetes mellitus due to underlying condition with ketoacidosis
    "E0810",
    "E0811",
    "E091",  # Drug or chemical induced diabetes mellitus with ketoacidosis
    "E0910",
    "E0911",
    "E101",  # Type 1 diabetes mellitus with ketoacidosis
    "E1010",
    "E1011",
    "E111",  # Type 2 diabetes mellitus with ketoacidosis
    "E1110",
    "E1111",
    "E131",  # Other specified diabetes mellitus with ketoacidosis
    "E1310",
    "E1311",
]

# Define CKD stage 5 codes
CKD5_CODE_V9 = [
    "40301",  # Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease
    "40311",  # Hypertensive chronic kidney disease, benign, with chronic kidney disease stage V or end stage renal disease
    "40391",  # Hypertensive chronic kidney disease, unspecified, with chronic kidney disease stage V or end stage renal disease
    "40402",
    "40403",
    "40412",
    "40413",
    "40492",
    "40493",
    "5855",  # Stage 5
    "5856",  # End stage renal disease
]
CKD5_CODE_V10 = [
    "I120",  # Hypertensive
    "I1311",
    "I132",
    "N185",  # stage 5
    "N186",  # End stage renal disease
]

TARGET_PATIENT_FILE = "target_patients.csv"

queryPostgresDf = PandaSQL(POSTGRESQL_CONNECTION_STRING)
