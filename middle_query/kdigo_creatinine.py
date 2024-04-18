import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from extract_mesurements import extractLabEventMesures
from patients import getTargetPatientIcu
from middle_query import SQL_PATH
from query_exceptions import ResultEmptyException


def extractKdigoCreatinine():
    OUTPUT_PATH = TEMP_PATH / "kdigo_creatinine.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfTargetPatients = getTargetPatientIcu()

    LABEVENT_FILE = "labevent_50912.csv"
    dfLabevent = extractLabEventMesures(50912, LABEVENT_FILE)
    dfLabevent["charttime"] = pd.to_datetime(dfLabevent["charttime"], format="ISO8601")

    result = pd.DataFrame()
    with open(SQL_PATH / "kdigo_creatinine.sql", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "labevents": dfLabevent,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
