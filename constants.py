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



TARGET_PATIENT_FILE = "target_patients.csv"

queryPostgresDf = PandaSQL(POSTGRESQL_CONNECTION_STRING)

LEARNING_DATA_FILE = TEMP_PATH / "learning_data.csv"