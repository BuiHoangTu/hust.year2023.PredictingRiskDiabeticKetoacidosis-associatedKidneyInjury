import pandas as pd
from constants import queryPostgresDf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from utils.extract_mesurements import extractChartEventMesures
from mimic_sql import SQL_FOLDER
from utils.query_exceptions import ResultEmptyException
from mimic_sql import urine_output
from mimic_sql import weight_durations


def runSql():
    UO_RATE_FILE = "urine_output_rate.csv"

    if (TEMP_PATH / UO_RATE_FILE).exists():
        return pd.read_csv(TEMP_PATH / UO_RATE_FILE, parse_dates=["charttime"])

    # bpm - heart rate
    dfChartevents220045 = extractChartEventMesures(220045, "chartevents_220045.csv")
    dfChartevents220045["charttime"] = pd.to_datetime(dfChartevents220045["charttime"])

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    dfUrineOutput = urine_output.extractUrineOutput()

    dfWeightDuration = weight_durations.runSql()

    result = pd.DataFrame()
    with open(SQL_FOLDER / "urine_output_rate.sql", "r") as queryStr:
        map = {
            "target_patients": dfTargetPatients,
            "chartevents": dfChartevents220045,
            "icustays": dfTargetPatients,
            "urine_output": dfUrineOutput,
            "weight_durations": dfWeightDuration,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / UO_RATE_FILE)

    return result
