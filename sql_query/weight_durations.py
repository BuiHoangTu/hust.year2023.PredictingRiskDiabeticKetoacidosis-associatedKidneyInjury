import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from extract_mesurements import extractChartEventMesures
from sql_query.query_exceptions import ResultEmptyException

WEIGHT_DURATIONS_FILE = "weight_durations.csv"


def extractWeight():
    WEIGHT_FILE = "chartevents_weight.csv"
    if (TEMP_PATH / WEIGHT_FILE).exists():
        dfCharteventsWeight = pd.read_csv(TEMP_PATH / WEIGHT_FILE)
        pass
    else:
        dfCharteventsWeight = extractChartEventMesures([226512, 224639], WEIGHT_FILE)
        pass
    dfCharteventsWeight["charttime"] = pd.to_datetime(dfCharteventsWeight["charttime"])

    dfTargetPatients = pd.read_csv(TEMP_PATH / "target_patients.csv")
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    result = pd.DataFrame()
    with open(SQL_PATH / "weight_durations.sql", "r") as queryStr:
        map = {
            "target_patients": dfTargetPatients,
            "chartevents": dfCharteventsWeight,
            "icustays": dfTargetPatients,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / WEIGHT_DURATIONS_FILE)
    
    return result