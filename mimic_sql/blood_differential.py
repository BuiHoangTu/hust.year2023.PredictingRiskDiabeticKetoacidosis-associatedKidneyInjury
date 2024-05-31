from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from mimic_sql import SQL_FOLDER, SqlWrapper
from utils.extract_mesurements import extractLabEventMesures


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.stem + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["charttime"])

    sqlWrapper = SqlWrapper(
        queryPostgresDf,
        sqlPath=SQL_FOLDER,
        sqlFileName=THIS_FILE.stem + ".sql",
    )

    CHART_EVENT_IDs = [
        51300,
        51301,
        51755,
        52069,
        52073,
        51199,
        51133,
        52769,
        52074,
        51253,
        52075,
        51218,
        51146,
        51244,
        51245,
        51254,
        51256,
        51143,
        52135,
        51251,
        51257,
        51146,
        51200,
    ]
    dfLabEvent = extractLabEventMesures(
        CHART_EVENT_IDs, "charted_" + THIS_FILE.name + ".csv"
    )
    sqlWrapper["labevents"] = dfLabEvent

    return sqlWrapper.runSQL()
