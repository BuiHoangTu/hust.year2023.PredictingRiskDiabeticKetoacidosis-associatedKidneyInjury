from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from extract_mesurements import extractInputEvents
from query_exceptions import ResultEmptyException

def runSql():
    THIS_FILE = Path(__file__)
    
    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH) 

    CHART_EVENT_IDs = [
        221289

    ]
    CHARTED_EVENT_FILE = "charted_crrt.csv"

    dfChartEvent = extractInputEvents(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")


    result = pd.DataFrame()
    queryStr = (Path(__file__).parent /  (THIS_FILE.stem + ".sql")).read_text()

    queryStr = queryStr.replace("%", "%%")
    map = {
            "inputevents": dfChartEvent,#copy ten bang vao day
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result