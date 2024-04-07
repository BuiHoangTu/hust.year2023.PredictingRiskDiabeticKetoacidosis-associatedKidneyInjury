import pandas as pd
from constants import queryPostgresDf

from constants import AKD_SQL_PATH, TEMP_PATH
from extract_mesurements import extractChartEventMesures
from akd_stage.query_exceptions import ResultEmptyException


def extractCrrt():
    CRRT_OUTPUT_FILE = "crrt.csv"

    if (TEMP_PATH / CRRT_OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / CRRT_OUTPUT_FILE)

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
    CHARTED_EVENT_FILE = "charted_crrt.csv"

    dfChartEventCrrt = extractChartEventMesures(CHART_EVENT_IDs, CHARTED_EVENT_FILE)

    dfChartEventCrrt["charttime"] = pd.to_datetime(dfChartEventCrrt["charttime"])

    result = pd.DataFrame()
    with open(AKD_SQL_PATH / "crrt.sql", "r") as queryStr:
        map = {
            "chartevents": dfChartEventCrrt,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / CRRT_OUTPUT_FILE)

    return result
