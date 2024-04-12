from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from query_exceptions import ResultEmptyException
from extract_target_patients import extractTargetPatients


def runSql():
    OUTPUT_FILE = "sapsii.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfPatients = extractTargetPatients()

    map = {
        "icustays": dfPatients,
        "chartevents": None,
        "admissions": None,
        "services": None,
        "diagnoses_icd": None,
        "bg": None,
        "ventilation": None,
        "gcs": None,
        "vitalsign": None,
        "urine_output": None,
        "chemistry": None,
        "complete_blood_count": None,
        "enzyme": None,
        "age": None,
    }
    result = queryPostgresDf(
        (Path(__file__).parent / "./sapsii.sql").read_text(),
        map,
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
