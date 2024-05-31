from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from utils.extract_mesurements import extractLabEventMesures
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    CHART_EVENT_IDs = [
        50862,
        50930,
        50976,
        50868,
        50882,
        51006,
        50893,
        50902,
        50912,
        50931,
        50983,
        50971,
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
