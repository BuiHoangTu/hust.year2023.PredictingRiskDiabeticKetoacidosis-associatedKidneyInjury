from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from utils.extract_mesurements import extractChartEventMesures
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["intime_hr", "outtime_hr"])

    queryStr = (THIS_FILE.parent / (THIS_FILE.stem + ".sql")).read_text()

    map = {
        "chartevents": extractChartEventMesures(220045, "charted_" + THIS_FILE.name + ".csv"),
        "icustays": getTargetPatientIcu(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
