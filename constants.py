# mimic PATH (in which there are hosp and icu data)
from pathlib import Path


MIMIC_PATH = Path("../mimiciv2.2/")

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

IMPORTANT_MESUREMENTS_ICU = {
    227519: "urine_output",
    224639: "weight",
    227457: "plt",
    220615: "creatinine"
}

PREPROCESSED_MESUREMENTS = ["sugar"]

IMPORTANT_MESUREMENTS_LABEVENT = {
    51006: "bun"    
}

# Define ICD-9/10 codes for DKA
DKA_CODE_V9 = [
    "24910", # Secondary diabetes mellitus with ketoacidosis
    "24911",
    "25010",
    "25011",
    "25012",
    "25013",
]
DKA_CODE_V10 = [
    "E81", # Diabetes mellitus due to underlying condition with ketoacidosis
    "E810",
    "E811",
    # "E091", # Drug or chemical induced diabetes mellitus with ketoacidosis
    # "E0910",
    # "E0911",
    "E101", # Type 1 diabetes mellitus with ketoacidosis
    "E1010",
    "E1011",
    "E111", # Type 2 diabetes mellitus with ketoacidosis
    "E1110",
    "E1111",
    "E131", # Other specified diabetes mellitus with ketoacidosis
    "E1310",
    "E1311",
]

# Define CKD stage 5 codes
CKD5_CODE_V9 = [
    "40301",
    "40311",
    "40391",
    "40402",
    "40403",
    "40412",
    "40413",
    "40492",
    "40493",
    "5855",
    "5856",
]
CKD5_CODE_V10 = [
    "N185",
    "N186",  # stage 5 and End stage renal disease
    "I120",
    "I1311",
    "I132",
]