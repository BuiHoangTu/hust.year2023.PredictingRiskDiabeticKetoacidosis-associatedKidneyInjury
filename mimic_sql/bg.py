from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from mimic_sql import SQL_FOLDER, SqlWrapper
from utils.extract_mesurements import extractLabEventMesures
from utils.extract_mesurements import extractChartEventMesures


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.stem + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["admittime"])

    else:
        sqlWrapper = SqlWrapper(
            queryPostgresDf,
            sqlPath=SQL_FOLDER,
            sqlFileName=THIS_FILE.stem + ".sql",
        )
        CHART_EVENT_IDs = [
            52033,
            50801,
            50802,
            50803,
            50804,
            50805,
            50806,
            50808,
            50809,
            50810,
            50811,
            50813,
            50814,
            50815,
            50816,
            50817,
            50818,
            50819,
            50820,
            50821,
            50822,
            50823,
            50824,
            50825,
            50807,
            220277,
            223835,
        ]
        dfLabEvent = extractLabEventMesures(
            CHART_EVENT_IDs, "lab_" + THIS_FILE.stem + ".csv"
        )
        sqlWrapper["labevents"] = dfLabEvent

        dfChartEvent = extractChartEventMesures(
            CHART_EVENT_IDs, "charted_" + THIS_FILE.stem + ".csv"
        )
        sqlWrapper["chartevents"] = dfChartEvent

        result = sqlWrapper.runSQL()
        return result
