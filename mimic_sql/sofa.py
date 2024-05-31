from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
from mimic_sql import (
    bg,
    chemistry,
    complete_blood_count,
    dobutamine,
    enzyme,
    epinephrine,
    gcs,
    icustay_hourly,
    urine_output_rate,
    vitalsign,
    norepinephrine,
    ventilation,
)
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    queryStr = (THIS_FILE.parent / (THIS_FILE.stem + ".sql")).read_text()

    map = {
        "icustay_hourly": icustay_hourly.runSql(),
        "icustays": getTargetPatientIcu(),
        "bg": bg.runSql(),
        "ventilation": ventilation.extractVentilation(),
        "vitalsign": vitalsign.runSql(),
        "gcs": gcs.runSql(),
        "enzyme": enzyme.runSql(),
        "chemistry": chemistry.runSql(),
        "complete_blood_count": complete_blood_count.runSql(),
        "urine_output_rate": urine_output_rate.runSql(),
        "epinephrine": epinephrine.runSql(),
        "norepinephrine": norepinephrine.runSql(),
        "dopamine": dobutamine.runSql(),
        "dobutamine": dobutamine.runSql(),
    }

    result = queryPostgresDf(queryStr, map)

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
