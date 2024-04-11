import pandas as pd
from constants import queryPostgresDf

from constants import VAR_INTERVENTION_PATH, TEMP_PATH
from akd_stage.query_exceptions import ResultEmptyException
from extract_mesurements import extractChartEventMesures


def extractOxygenDelivery():
    OUTPUT_FILE = "oxygen_delivery.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    CHARTED_IDs = [223834, 227582, 227287, 226732]
    CHARTED_FILE = "chartevent_oxygen_delivery.csv"
    dfChartEvent = extractChartEventMesures(CHARTED_IDs, CHARTED_FILE)

    result = pd.DataFrame()
    with open(VAR_INTERVENTION_PATH / "mechanical_ventilation" / "oxygen_delivery.sql", "r") as queryStr:
        map = {
            "chartevents": dfChartEvent,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
