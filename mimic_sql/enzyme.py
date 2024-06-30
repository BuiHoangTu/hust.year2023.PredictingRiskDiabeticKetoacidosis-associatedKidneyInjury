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
        50861 ,
        50863 , 
        50878 , 
        50867 , 
        50885 , 
        50884 , 
        50883 , 
        50910 , 
        50911 , 
        50927 , 
        50954 ,
    ]

    dfChartEvent = extractLabEventMesures(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")

    #dfChartEventCrrt["charttime"] = pd.to_datetime(dfChartEventCrrt["charttime"])

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
