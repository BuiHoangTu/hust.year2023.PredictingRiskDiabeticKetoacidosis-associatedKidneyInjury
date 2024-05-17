from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from utils.query_exceptions import ResultEmptyException
import mimic_sql.gcs as gcs


def runSql():
    GCS_OUTPUT_PATH = TEMP_PATH / "first_day_gcs.csv"

    if (GCS_OUTPUT_PATH).exists():
        return pd.read_csv(GCS_OUTPUT_PATH)

    dfPatient = getTargetPatientIcu()

    queryStr = (Path(__file__).parent / "first_day_gcs.sql").read_text()
    map = {
        "icustays": dfPatient,
        "gcs": gcs.runSql(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(GCS_OUTPUT_PATH)

    return result
