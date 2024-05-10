import pandas as pd
from constants import queryPostgresDf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from mimic_sql import SQL_FOLDER
from mimic_sql import crrt
from mimic_sql.kdigo_creatinine import extractKdigoCreatinine
from mimic_sql.kdigo_uo import extractKdigoUrineOutput
from utils.query_exceptions import ResultEmptyException


def extractKdigoStages():
    OUTPUT_FILE = "kdigo_stages.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE, parse_dates=["charttime"])

    dfKdigoCreat = extractKdigoCreatinine()

    dfKdigoUO = extractKdigoUrineOutput()

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfCrrt = crrt.runSql()

    result = pd.DataFrame()
    with open(SQL_FOLDER / "kdigo_stages.sql", "r") as queryStr:
        map = {
            "kdigo_creatinine": dfKdigoCreat,
            "kdigo_uo": dfKdigoUO,
            "icustays": dfTargetPatients,
            "crrt": dfCrrt,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
