from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from utils.extract_mesurements import extractChartEventMesures
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    CHART_EVENT_IDs = [
        223901,
        223900,
        220739,
    ]

    dfChartEvents = extractChartEventMesures(CHART_EVENT_IDs, "charted_" + THIS_FILE.name + ".csv")

    queryStr = (Path(__file__).parent /  (THIS_FILE.stem + ".sql")).read_text()
    map = {
        "chartevents": dfChartEvents,
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
