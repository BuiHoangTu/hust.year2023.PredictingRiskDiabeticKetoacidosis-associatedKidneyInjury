from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from utils.query_exceptions import ResultEmptyException
from utils.extract_mesurements import extractChartEventMesures


def extractOxygenDelivery():
    OUTPUT_FILE = "oxygen_delivery.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE, parse_dates=["charttime"])

    CHARTED_IDs = [223834, 227582, 227287, 226732]
    CHARTED_FILE = "chartevent_oxygen_delivery.csv"
    dfChartEvent = extractChartEventMesures(CHARTED_IDs, CHARTED_FILE)

    result = pd.DataFrame()
    with open(Path(__file__).parent / "oxygen_delivery.sql", "r") as queryStr:
        map = {
            "chartevents": dfChartEvent,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
