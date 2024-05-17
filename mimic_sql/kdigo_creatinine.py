import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from utils.extract_mesurements import extractLabEventMesures
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from mimic_sql import SQL_FOLDER
from utils.query_exceptions import ResultEmptyException


def extractKdigoCreatinine():
    OUTPUT_PATH = TEMP_PATH / "kdigo_creatinine.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    dfTargetPatients = getTargetPatientIcu()

    LABEVENT_FILE = "labevent_50912.csv"
    dfLabevent = extractLabEventMesures(50912, LABEVENT_FILE)
    dfLabevent["charttime"] = pd.to_datetime(dfLabevent["charttime"], format="ISO8601")

    result = pd.DataFrame()
    with open(SQL_FOLDER / "kdigo_creatinine.sql", "r") as queryStr:
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
