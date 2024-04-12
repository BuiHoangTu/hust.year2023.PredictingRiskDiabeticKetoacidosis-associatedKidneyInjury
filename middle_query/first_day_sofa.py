from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from extract_target_patients import extractTargetPatients
from middle_query import first_day_gcs, first_day_urine_output, first_day_vitalsign, ventilation
from query_exceptions import ResultEmptyException
from middle_query import first_day_lab


def runSql():
    GCS_OUTPUT_PATH = TEMP_PATH / "first_day_sofa.csv"

    if (GCS_OUTPUT_PATH).exists():
        return pd.read_csv(GCS_OUTPUT_PATH)

    queryStr = (Path(__file__).parent / "./first_day_sofa.sql").read_text()
    map = {
        "icustays": extractTargetPatients(),
        "norepinephrine": None,
        "epinephrine": None,
        "dobutamine": None,
        "dopamine": None,
        "bg": None,
        "ventilation": ventilation.extractVentilation(),
        "first_day_vitalsign": first_day_vitalsign.runSql(),
        "first_day_lab": first_day_lab.runSql(),
        "first_day_urine_output": first_day_urine_output.runSql(),
        "first_day_gcs": first_day_gcs.runSql(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(GCS_OUTPUT_PATH)

    return result
