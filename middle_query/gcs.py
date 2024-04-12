from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from extract_mesurements import extractChartEventMesures
from query_exceptions import ResultEmptyException


def runSql():
    GCS_OUTPUT_PATH = TEMP_PATH / "gcs.csv"

    if (GCS_OUTPUT_PATH).exists():
        return pd.read_csv(GCS_OUTPUT_PATH)

    CHART_EVENT_IDs = [
        223901,
        223900,
        220739,
    ]

    dfChartEvents = extractChartEventMesures(CHART_EVENT_IDs, "charted_gcs.csv")

    queryStr = (Path(__file__).parent / "gcs.sql").read_text()
    map = {
        "chartevents": dfChartEvents,
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(GCS_OUTPUT_PATH)

    return result
