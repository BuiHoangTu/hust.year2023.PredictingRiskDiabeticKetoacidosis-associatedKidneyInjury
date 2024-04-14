from pathlib import Path
import pandas as pd

from constants import MIMIC_PATH, TEMP_PATH, queryPostgresDf
from extract_target_patients import extractTargetPatients
from middle_query import age, first_day_gcs, first_day_urine_output, first_day_vitalsign
from query_exceptions import ResultEmptyException
from middle_query.ventilation import extractVentilation


def runSql():
    OUTPUT_PATH = TEMP_PATH / "oasis.csv"

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfServices = pd.read_csv(MIMIC_PATH / "hosp" / "services.csv")
    dfServices["transfertime"] = pd.to_datetime(dfServices["transfertime"])

    dfVent = extractVentilation()
    dfVent["starttime"] = pd.to_datetime(dfVent["starttime"])
    dfVent["endtime"] = pd.to_datetime(dfVent["endtime"])

    dfAdmission = pd.read_csv(MIMIC_PATH / "hosp" / "admissions.csv")
    dfAdmission["admittime"] = pd.to_datetime(dfAdmission["admittime"])
    dfAdmission["deathtime"] = pd.to_datetime(dfAdmission["deathtime"])
    dfAdmission["dischtime"] = pd.to_datetime(dfAdmission["dischtime"])

    queryStr = (Path(__file__).parent / "oasis.sql").read_text()

    # escape string
    queryStr = queryStr.replace("%", "%%")

    map = {
        "icustays": extractTargetPatients(),
        "services": dfServices,
        "ventilation": dfVent,
        "admissions": dfAdmission,
        "patients": extractTargetPatients(),
        "age": age.runSql(),
        "first_day_gcs": first_day_gcs.runSql(),
        "first_day_vitalsign": first_day_vitalsign.runSql(),
        "first_day_urine_output": first_day_urine_output.runSql(), 
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
