import pandas as pd
from constants import queryPostgresDf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from middle_query import SQL_PATH
from middle_query.crrt import extractCrrt
from middle_query.kdigo_creatinine import extractKdigoCreatinine
from middle_query.kdigo_uo import extractKdigoUrineOutput
from query_exceptions import ResultEmptyException


def extractKdigoStages():
    OUTPUT_FILE = "kdigo_stages.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfKdigoCreat = extractKdigoCreatinine()

    dfKdigoUO = extractKdigoUrineOutput()

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfCrrt = extractCrrt()

    result = pd.DataFrame()
    with open(SQL_PATH / "kdigo_stages.sql", "r") as queryStr:
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
