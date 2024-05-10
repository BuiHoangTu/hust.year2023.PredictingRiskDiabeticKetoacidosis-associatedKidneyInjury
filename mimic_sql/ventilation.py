from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from utils.query_exceptions import ResultEmptyException
from mimic_sql.oxygen_delivery import extractOxygenDelivery
from mimic_sql.ventilator_setting import extractVentilatorSetting


def extractVentilation():
    OUTPUT_FILE = "ventilation.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE, parse_dates=["starttime", "endtime"])

    dfVentSetting = extractVentilatorSetting()
    dfVentSetting["charttime"] = pd.to_datetime(dfVentSetting["charttime"], format="ISO8601")

    dfOxygen = extractOxygenDelivery()
    dfOxygen["charttime"] = pd.to_datetime(dfOxygen["charttime"], format="ISO8601")

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    result = pd.DataFrame()
    with open(Path(__file__).parent/ "ventilation.sql", "r") as queryStr:
        map = {
            "ventilator_setting": dfVentSetting,
            "oxygen_delivery": dfOxygen,
            "target_patients": dfTargetPatients,
            "icustays": dfTargetPatients,
        }

        result = queryPostgresDf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
