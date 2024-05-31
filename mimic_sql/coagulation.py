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

    CHART_EVENT_IDs = [# copy tu ben sql
        51196,
        51214,
        51297,
        51237,
        51274,
        51275,
    ]

    dfChartEvent = extractLabEventMesures(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")


    result = pd.DataFrame()
    queryStr = (Path(__file__).parent /  (THIS_FILE.stem + ".sql")).read_text()
    map = {
            "labevents": dfChartEvent,#copy ten bang vao day
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
