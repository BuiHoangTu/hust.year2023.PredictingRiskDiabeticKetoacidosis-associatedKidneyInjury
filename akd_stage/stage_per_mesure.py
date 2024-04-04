import pandas as pd
from pandasql import sqldf

from constants import AKD_SQL_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from mark_akd_creatinine import markAkdCreatinine
from akd_stage.query_exceptions import ResultEmptyException
from akd_stage.urine_output_rate import extractOURate


def extractAkdPerMesure():
    MESURE_STAGE_FILE = "stage_per_mesure.csv"

    if (TEMP_PATH / MESURE_STAGE_FILE).exists():
        return pd.read_csv(TEMP_PATH / MESURE_STAGE_FILE)

    dfCreatStg = markAkdCreatinine()

    dfUoRate = extractOURate()
    # convert columns' names between 2 sql
    dfUoRate["uo_rt_6hr"] = dfUoRate["uo_mlkghr_6hr"]
    dfUoRate["uo_rt_12hr"] = dfUoRate["uo_mlkghr_12hr"]
    dfUoRate["uo_rt_24hr"] = dfUoRate["uo_mlkghr_24hr"]

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    result = pd.DataFrame()
    with open(AKD_SQL_PATH / "stage_per_mesure.sql", "r") as queryStr:
        map = {
            "target_patients": dfTargetPatients,
            "kdigo_creat": dfCreatStg,
            "icustays": dfTargetPatients,
            "kdigo_uo": dfUoRate,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / MESURE_STAGE_FILE)

    return result
