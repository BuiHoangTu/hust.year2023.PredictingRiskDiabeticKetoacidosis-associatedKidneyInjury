import pandas as pd
from pandasql import sqldf

from constants import SQL_PATH, TEMP_PATH
from mark_akd_creatinine import markAkdCreatinine
from sql_query.query_exceptions import ResultEmptyException
from sql_query.urine_output_rate import extractOURate


def extractAkd():
    AKD_FILE = "akd.csv"

    if (TEMP_PATH / AKD_FILE).exists():
        return pd.read_csv(TEMP_PATH / AKD_FILE)

    dfCreatStg = markAkdCreatinine()

    dfUoRate = extractOURate()

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
