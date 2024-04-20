import pandas as pd
from constants import MIMIC_PATH, queryPostgresDf

from constants import TEMP_PATH
from middle_query import SQL_PATH
from query_exceptions import ResultEmptyException


def runSql():
    OUTPUT_PATH = TEMP_PATH / "age.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["admittime"])

    queryStr = (SQL_PATH / "./age.sql").read_text()

    result = queryPostgresDf(
        queryStr,
        {
            "admissions": pd.read_csv(
                MIMIC_PATH / "hosp" / "admissions.csv", parse_dates=["admittime"]
            ),
            "patients": pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv"),
        },
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
