from pathlib import Path
from pandasql import PandaSQL
from secret import MIMIC_PATH_STR, POSTGRESQL_CONNECTION_STRING


MIMIC_PATH = Path(MIMIC_PATH_STR)

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# measures whose null represent false value
NULLABLE_MEASURES = [
    "dka_type",
    "macroangiopathy",
    "microangiopathy",
    "mechanical_ventilation",
    "use_NaHCO3",
    "history_aci",
    "history_ami",
    "congestive_heart_failure",
    "liver_disease",
    "ckd_stage",
    "malignant_cancer",
    "hypertension",
    "uti",
    "chronic_pulmonary_disease",
]

# categorical values
CATEGORICAL_MEASURES = [
    "dka_type",
    "gender",
    "race",
    "liver_disease",
    "ckd_stage",
]


TARGET_PATIENT_FILE = "target_patients.csv"

queryPostgresDf = PandaSQL(POSTGRESQL_CONNECTION_STRING)

## Archived notebooks
ARCHIVED_NOTEBOOKS_PATH = Path("archived_notebooks")
ARCHIVED_NOTEBOOKS_PATH.mkdir(parents=True, exist_ok=True)