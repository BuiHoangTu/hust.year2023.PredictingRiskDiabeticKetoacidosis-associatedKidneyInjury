import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from sql_query.query_exceptions import ResultEmptyException
from sql_query.urine_output import extractUrineOutput
from sql_query.weight_durations import extractWeightDuration


def extractKdigoUrineOutput():
    OUTPUT_FILE = "kdigo_uo.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)


    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfUO = extractUrineOutput() 

    dfWeightDuration = extractWeightDuration() 

    result = pd.DataFrame()
    with open(SQL_PATH / "kdigo_uo.sql", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "urine_output": dfUO,
            "weight_durations": dfWeightDuration,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
