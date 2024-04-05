import pandas as pd
from pandasql import sqldf

from constants import VAR_INTERVENTION_PATH, TEMP_PATH
from akd_stage.query_exceptions import ResultEmptyException
from extract_mesurements import extractChartEventMesures


def extractVentilatorSetting():
    OUTPUT_FILE = "ventilator_setting.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

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
    with open(VAR_INTERVENTION_PATH / "ventilator_setting.sql", "r") as queryStr:
        map = {
            "chartevents": dfChartevent,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
