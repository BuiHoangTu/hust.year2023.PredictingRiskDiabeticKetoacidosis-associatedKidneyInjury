from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from patients import getTargetPatientIcu
from middle_query import vitalsign
from query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfVitalSign = vitalsign.runSql()
    dfVitalSign["charttime"] = pd.to_datetime(dfVitalSign["charttime"])

    queryStr = (Path(__file__).parent / (THIS_FILE.stem + ".sql")).read_text()
    result = queryPostgresDf(
        queryStr,
        {
            "vitalsign": dfVitalSign,
            "icustays": getTargetPatientIcu(),
        },
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    df = result
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()
