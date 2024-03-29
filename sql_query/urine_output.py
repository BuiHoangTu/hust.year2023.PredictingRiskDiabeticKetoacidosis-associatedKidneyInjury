import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from extract_mesurements import extractOutputEvents
from sql_query.query_exceptions import ResultEmptyException

URINE_OUTPUT_FILE = "urine_output.csv"


def extractUrineOutput():
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
    CHARTED_URINE_FILE = "urine_output.csv"
    if (TEMP_PATH / CHARTED_URINE_FILE).exists():
        dfOutputeventsUrine = pd.read_csv(TEMP_PATH / CHARTED_URINE_FILE)
        pass
    else:
        dfOutputeventsUrine = extractOutputEvents(OUTPUT_EVENT_URINE_IDs, CHARTED_URINE_FILE)
        pass
    dfOutputeventsUrine["charttime"] = pd.to_datetime(dfOutputeventsUrine["charttime"])

    result = pd.DataFrame()
    with open(SQL_PATH / "urine_output.sql", "r") as queryStr:
        map = {
            "outputevents": dfOutputeventsUrine,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / URINE_OUTPUT_FILE)

    return result
