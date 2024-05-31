import pandas as pd
from constants import queryPostgresDf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from mimic_sql import SQL_FOLDER
from utils.query_exceptions import ResultEmptyException
from mimic_sql.urine_output import extractUrineOutput
from mimic_sql.weight_durations import runSql


def extractKdigoUrineOutput():
    OUTPUT_FILE = "kdigo_uo.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE, parse_dates=["charttime"])

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfUO = extractUrineOutput() 

    dfWeightDuration = runSql() 

    result = pd.DataFrame()
    with open(SQL_FOLDER / "kdigo_uo.sql", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "urine_output": dfUO,
            "weight_durations": dfWeightDuration,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
