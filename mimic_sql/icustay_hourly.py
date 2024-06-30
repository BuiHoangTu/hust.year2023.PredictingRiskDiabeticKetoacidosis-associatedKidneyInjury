from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from mimic_sql import icustay_times
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["endtime"])

    queryStr = (THIS_FILE.parent / (THIS_FILE.stem + ".sql")).read_text()

    map = {
        "icustay_times": icustay_times.runSql(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
