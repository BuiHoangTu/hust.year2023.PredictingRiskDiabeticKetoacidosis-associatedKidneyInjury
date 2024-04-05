import pandas as pd
from pandasql import sqldf

from constants import VAR_INTERVENTION_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from akd_stage.query_exceptions import ResultEmptyException
from variables_interventions.oxygen_delivery import extractOxygenDelivery
from variables_interventions.ventilator_setting import extractVentilatorSetting


def extractMechVent():
    OUTPUT_FILE = "ventilation.csv"

    if (TEMP_PATH / OUTPUT_FILE).exists():
        return pd.read_csv(TEMP_PATH / OUTPUT_FILE)

    dfVentSetting = extractVentilatorSetting()

    dfOxygen = extractOxygenDelivery()

    dfTargetPatients = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatients["intime"] = pd.to_datetime(dfTargetPatients["intime"])
    dfTargetPatients["outtime"] = pd.to_datetime(dfTargetPatients["outtime"])

    result = pd.DataFrame()
    with open(VAR_INTERVENTION_PATH / "urine_output_rate.sql", "r") as queryStr:
        map = {
            "ventilator_setting": dfVentSetting,
            "oxygen_delivery": dfOxygen,
            "target_patients": dfTargetPatients,
            "icustays": dfTargetPatients,
        }

        result = sqldf(queryStr.read(), map)
        pass

    if result is None:
        raise ResultEmptyException()

    result["mechanical_ventilation"] = result["ventilation_status"].isin(
        [
            "Tracheostomy",
            "InvasiveVent",
        ]
    )
    result.to_csv(TEMP_PATH / OUTPUT_FILE)

    return result
