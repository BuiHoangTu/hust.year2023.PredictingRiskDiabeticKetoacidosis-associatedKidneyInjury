import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from sql_query.query_exceptions import ResultEmptyException
from sql_query.stage_per_mesure import extractAkdPerMesure


def extractAkdPerPatient():
    PATIENT_STAGE_FILE = "stage_per_patient.csv"

    if (TEMP_PATH / PATIENT_STAGE_FILE).exists():
        return pd.read_csv(TEMP_PATH / PATIENT_STAGE_FILE)

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfStagePerMesure = extractAkdPerMesure()
    dfStagePerMesure["creat"] = dfStagePerMesure["value"]

    result = pd.DataFrame()
    with open(SQL_PATH / "stage_per_patient.sql", "r") as queryStr:
        map = {
            "target_patients": dfTargetPatients,
            "kdigo_stages": dfStagePerMesure,
            "icustays": dfTargetPatients,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / PATIENT_STAGE_FILE)

    return result
