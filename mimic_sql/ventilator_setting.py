from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from utils.query_exceptions import ResultEmptyException
from utils.extract_mesurements import extractChartEventMesures


def extractVentilatorSetting():
    OUTPUT_FILE = "ventilator_setting.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE, parse_dates=["charttime"])

    CHARTEVENT_IDs = [
        224688,
        224689,
        224690,
        224687,
        224685,
        224684,
        224686,
        224696,
        220339,
        224700,
        223835,
        223849,
        229314,
        223848,
        224691,
    ]
    CHARTED_FILE = "chartevent_ventilator_setting.csv"
    dfChartevent = extractChartEventMesures(CHARTEVENT_IDs, CHARTED_FILE)

    result = pd.DataFrame()
    with open(
        Path(__file__).parent / "ventilator_setting.sql", "r"
    ) as queryStr:
        map = {
            "chartevents": dfChartevent,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
