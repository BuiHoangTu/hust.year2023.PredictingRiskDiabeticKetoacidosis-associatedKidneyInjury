import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from mimic_sql import SQL_FOLDER
from mimic_sql.urine_output import extractUrineOutput
from utils.query_exceptions import ResultEmptyException


def runSql():
    OUTPUT_PATH = TEMP_PATH / "./first_day_urine_output.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfUO = extractUrineOutput()
    dfUO["charttime"] = pd.to_datetime(dfUO["charttime"])

    queryStr = (SQL_FOLDER / "./first_day_urine_output.sql").read_text()
    result = queryPostgresDf(
        queryStr,
        {
            "urine_output": dfUO,
            "icustays": getTargetPatientIcu(),
        },
    )

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
