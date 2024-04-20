from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from extract_mesurements import extractLabEventMesures
from extract_mesurements import extractChartEventMesures
from query_exceptions import ResultEmptyException

def runSql():
    THIS_FILE = Path(__file__)
    
    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH) 

    CHART_EVENT_IDs = [
        52033,50801,50802,50803,50804,50805,50806,50808,50809,50810,50811,50813,50814,50815,
        50816,50817,50818,50819,50820,50821,50822,50823,50824,50825,50807,220277,223835
    ]
    CHARTED_EVENT_FILE = "charted_crrt.csv"

    dfLabEvent = extractLabEventMesures(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")
    dfChartEvent = extractChartEventMesures(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")


    result = pd.DataFrame()
    queryStr = (Path(__file__).parent /  (THIS_FILE.stem + ".sql")).read_text()

    queryStr = queryStr.replace("%", "%%")
    map = {
            "labevents": dfLabEvent,#copy ten bang vao day
            "chartevents": dfChartEvent,
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result