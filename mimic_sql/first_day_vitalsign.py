import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from mimic_sql import SQL_FOLDER, vitalsign
from utils.query_exceptions import ResultEmptyException


def runSql():
    OUTPUT_PATH = TEMP_PATH / "first_day_vitalsign.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfVitalSign = vitalsign.runSql()
    dfVitalSign["charttime"] = pd.to_datetime(dfVitalSign["charttime"])

    result = queryPostgresDf(
        (SQL_FOLDER / "./first_day_vitalsign.sql").read_text(),
        {
            "vitalsign": dfVitalSign,
            "icustays": getTargetPatientIcu(),
        },
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
