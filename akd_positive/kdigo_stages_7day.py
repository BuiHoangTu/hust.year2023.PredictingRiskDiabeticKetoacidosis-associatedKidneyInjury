from pathlib import Path
import pandas as pd
from pandasql import sqldf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from mimic_sql.kdigo_stages import extractKdigoStages
from utils.query_exceptions import ResultEmptyException


def extractKdigoStages7day():
    OUTPUT_FILE = "kdigo_stages_7day.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfKdigoStage = extractKdigoStages()

    result = pd.DataFrame()
    with open(Path(__file__).parent / "kdigo_stages_7day.sqlite", "r") as queryStr:
        map = {
            "icustays": dfTargetPatients,
            "kdigo_stages": dfKdigoStage,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
