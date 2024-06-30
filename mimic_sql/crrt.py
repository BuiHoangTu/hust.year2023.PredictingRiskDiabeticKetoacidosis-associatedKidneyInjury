import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from utils.extract_mesurements import extractChartEventMesures
from mimic_sql import SQL_FOLDER
from utils.query_exceptions import ResultEmptyException


def runSql():
    CRRT_OUTPUT_PATH = TEMP_PATH / "crrt.csv"

    if (CRRT_OUTPUT_PATH).exists():
        return pd.read_csv(CRRT_OUTPUT_PATH, parse_dates=["charttime"])

    CHART_EVENT_IDs = [
        227290,
        224146,
        224149,
        224144,
        228004,
        225183,
        225977,
        224154,
        224151,
        224150,
        225958,
        224145,
        224191,
        228005,
        228006,
        225976,
        224153,
        224152,
        226457,
    ]

    dfChartEventCrrt = extractChartEventMesures(CHART_EVENT_IDs, "charted_crrt.csv")

    queryStr = (SQL_FOLDER / "crrt.sql").read_text()
    map = {
        "chartevents": dfChartEventCrrt,
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(CRRT_OUTPUT_PATH)

    return result
