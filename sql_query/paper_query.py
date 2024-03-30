import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from extract_mesurements import extractChartEventMesures
from mark_akd_creatinine import LAB_CREAT_STAGE_FILE_NAME
from sql_query.query_exceptions import ResultEmptyException
from sql_query.urine_output_rate import UO_RATE_FILE, extractOURate

AKD_FILE = "akd.csv"


def extractAkd():
    MARKED_CREAT_FILE = LAB_CREAT_STAGE_FILE_NAME

    dfCreatStg = extractChartEventMesures(220045, MARKED_CREAT_FILE)

    if (TEMP_PATH / UO_RATE_FILE).exists():
        dfUoRate = pd.read_csv(TEMP_PATH / UO_RATE_FILE)
        pass
    else:
        dfUoRate = extractOURate()
        pass

    dfTargetPatients = pd.read_csv(TEMP_PATH / "target_patients.csv")
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    result = pd.DataFrame()
    with open(SQL_PATH / "paper_query.sql", "r") as queryStr:
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
    result.to_csv(TEMP_PATH / AKD_FILE)

    return result
