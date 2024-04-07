import pandas as pd
from constants import queryPostgresDf

from constants import AKD_SQL_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from extract_mesurements import extractLabEventMesures
from akd_stage.query_exceptions import ResultEmptyException


def extractKdigoCreatinine():
    OUTPUT_FILE = "kdigo_creatinine.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    LABEVENT_FILE = "labevent_50912.csv"
    dfLabevent = extractLabEventMesures(50912, LABEVENT_FILE)

    result = pd.DataFrame()
    with open(AKD_SQL_PATH / "kdigo_creatinine.sql", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "labevents": dfLabevent,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
