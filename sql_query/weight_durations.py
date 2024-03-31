import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from extract_mesurements import extractChartEventMesures
from sql_query.query_exceptions import ResultEmptyException


def extractWeight():
    WEIGHT_DURATIONS_FILE = "weight_durations.csv"
    
    if (TEMP_PATH / WEIGHT_DURATIONS_FILE).exists():
        return pd.read_csv(TEMP_PATH / WEIGHT_DURATIONS_FILE)
    
    dfCharteventsWeight = extractChartEventMesures([226512, 224639], "chartevents_weight.csv")

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
