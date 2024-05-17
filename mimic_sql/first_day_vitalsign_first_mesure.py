from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from mimic_sql import vitalsign
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.stem + ".csv")

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

    df = result
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()
    
    df.to_csv(OUTPUT_PATH)
    return df
