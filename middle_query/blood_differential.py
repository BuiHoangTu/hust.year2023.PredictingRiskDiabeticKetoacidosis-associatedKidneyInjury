from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from extract_mesurements import extractLabEventMesures
from query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    CHART_EVENT_IDs = [
        51300,
        51301,
        51755,
        52069,
        52073,
        51199,
        51133,
        52769,
        52074,
        51253,
        52075,
        51218,
        51146,
        51244,
        51245,
        51254,
        51256,
        51143,
        52135,
        51251,
        51257,
        51146,
        51200,
    ]

    dfChartEvent = extractLabEventMesures(
        CHART_EVENT_IDs, "charted_" + THIS_FILE.name + ".csv"
    )

    result = pd.DataFrame()
    queryStr = (Path(__file__).parent / (THIS_FILE.stem + ".sql")).read_text()

    queryStr = queryStr.replace("%", "%%")
    map = {
        "labevents": dfChartEvent,  # copy ten bang vao day
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
