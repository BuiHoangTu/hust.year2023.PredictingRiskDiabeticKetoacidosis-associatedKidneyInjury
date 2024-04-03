import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from sql_query.kdigo_stages import extractKdigoStages
from sql_query.query_exceptions import ResultEmptyException


def extractKdigoStages7day():
    OUTPUT_FILE = "kdigo_stages_7day.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfKdigoStage = extractKdigoStages()

    result = pd.DataFrame()
    with open(SQL_PATH / "kdigo_stages_7day.sql", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "kdigo_stages": dfKdigoStage,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
