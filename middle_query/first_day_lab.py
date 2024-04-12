from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from query_exceptions import ResultEmptyException
from extract_target_patients import extractTargetPatients


def runSql():
    CRRT_OUTPUT_FILE = "first_day_lab.csv"

    if (TEMP_PATH / CRRT_OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / CRRT_OUTPUT_FILE)

    dfPatients = extractTargetPatients()

    map = {
        "icustays": dfPatients,
        "complete_blood_count": None,
        "chemistry": None,
        "blood_differential": None,
        "coagulation": None,
        "enzyme": None,
    }
    result = queryPostgresDf((Path(__file__).parent / "first_day_lab_sql" / "first_day_lab.sql").read_text(), map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / CRRT_OUTPUT_FILE)

    return result
