import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from utils.extract_mesurements import extractChartEventMesures
from mimic_sql import SQL_FOLDER
from utils.query_exceptions import ResultEmptyException


def runSql():
    OUTPUT_PATH = TEMP_PATH / "./vitalsign.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    CHARTED_IDs = [
        220045,
        225309,
        225310,
        225312,
        220050,
        220051,
        220052,
        220179,
        220180,
        220181,
        220210,
        224690,
        220277,
        225664,
        220621,
        226537,
        223762,
        223761,
        224642,
    ]

    dfChartevent = extractChartEventMesures(CHARTED_IDs, "charted_vitalsign.csv")

    queryStr = (SQL_FOLDER / "./vitalsign.sql").read_text()
    result = queryPostgresDf(
        queryStr,
        {
            "chartevents": dfChartevent,
        },
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
