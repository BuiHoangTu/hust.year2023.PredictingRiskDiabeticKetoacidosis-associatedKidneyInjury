import pandas as pd
from constants import queryPostgresDf

from constants import AKD_SQL_PATH, TEMP_PATH
from extract_mesurements import extractOutputEvents
from query_exceptions import ResultEmptyException


def extractUrineOutput():
    URINE_OUTPUT_FILE = "urine_output.csv"
    
    if (TEMP_PATH / URINE_OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / URINE_OUTPUT_FILE)

    OUTPUT_EVENT_URINE_IDs = [
        226559,
        226560,
        226561,
        226584,
        226563,
        226564,
        226565,
        226567,
        226557,
        226558,
        227488,
        227489,
    ]
    CHARTED_URINE_FILE = "urine_mesures.csv"

    dfOutputeventsUrine = extractOutputEvents(OUTPUT_EVENT_URINE_IDs, CHARTED_URINE_FILE)

    dfOutputeventsUrine["charttime"] = pd.to_datetime(dfOutputeventsUrine["charttime"])

    result = pd.DataFrame()
    with open(AKD_SQL_PATH / "urine_output.sql", "r") as queryStr:
        map = {
            "outputevents": dfOutputeventsUrine,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / URINE_OUTPUT_FILE)

    return result
