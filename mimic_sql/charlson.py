from pathlib import Path
import pandas as pd

from constants import TEMP_PATH, queryPostgresDf
from mimic_sql import age
from notebook_wrappers.target_patients_wrapper import getTargetPatientAdmission, getTargetPatientIcd
from utils.query_exceptions import ResultEmptyException


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)
    

    result = pd.DataFrame()
    queryStr = (Path(__file__).parent / (THIS_FILE.stem + ".sql")).read_text()

    queryStr = queryStr.replace("%", "%%")
    map = {
        "diagnoses_icd": getTargetPatientIcd(), 
        "admissions": getTargetPatientAdmission(),
        "age": age.runSql(),
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
