import pandas as pd

from constants import TEMP_PATH, VAR_SCORE_SYS_PATH, queryPostgresDf
from extract_target_patients import extractTargetPatients
from query_exceptions import ResultEmptyException
from variables_scoring_systems.gsc.gcs import extractAllGcs


def extractFisrtDayGcs():
    GCS_OUTPUT_PATH = TEMP_PATH / "first_day_gcs.csv"

    if (GCS_OUTPUT_PATH).exists():
        return pd.read_csv(GCS_OUTPUT_PATH)

    dfPatient = extractTargetPatients()

    queryStr = (VAR_SCORE_SYS_PATH / "gsc" / "first_day_gcs.sql").read_text()
    map = {
        "icustays": dfPatient,
        "gcs": extractAllGcs(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(GCS_OUTPUT_PATH)

    return result
