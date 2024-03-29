import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from extract_mesurements import extractChartEventMesures
from sql_query.query_exceptions import ResultEmptyException
from sql_query.urine_output import URINE_OUTPUT_FILE
from sql_query.weight_durations import WEIGHT_DURATIONS_FILE, extractWeight

UO_RATE_FILE = "urine_output_rate.csv"


def extractOURate():
    CHARTED_220045_FILE = "chartevents_220045.csv"
    if (TEMP_PATH / CHARTED_220045_FILE).exists():
        dfChartevents220045 = pd.read_csv(TEMP_PATH / CHARTED_220045_FILE)
        pass
    else:
        dfChartevents220045 = extractChartEventMesures(220045, CHARTED_220045_FILE)
        pass
    dfChartevents220045["charttime"] = pd.to_datetime(dfChartevents220045["charttime"])

    dfTargetPatients = pd.read_csv(TEMP_PATH / "target_patients.csv")
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    if (TEMP_PATH / URINE_OUTPUT_FILE).exists():
        dfUrineOutput = pd.read_csv(TEMP_PATH / URINE_OUTPUT_FILE)
        pass
    else:
        dfUrineOutput = extractChartEventMesures(220045, URINE_OUTPUT_FILE)
        pass

    if (TEMP_PATH / WEIGHT_DURATIONS_FILE).exists():
        dfWeightDuration = pd.read_csv(TEMP_PATH / WEIGHT_DURATIONS_FILE)
        pass
    else:
        dfWeightDuration = extractWeight()
        pass

    result = pd.DataFrame()
    with open(SQL_PATH / "urine_output_rate.sql", "r") as queryStr:
        map = {
            "target_patients": dfTargetPatients,
            "chartevents": dfChartevents220045,
            "icustays": dfTargetPatients,
            "urine_output": dfUrineOutput,
            "weight_durations": dfWeightDuration
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / UO_RATE_FILE)

    return result
