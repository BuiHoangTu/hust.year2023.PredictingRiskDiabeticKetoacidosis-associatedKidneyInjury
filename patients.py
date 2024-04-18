import nbformat
import pandas as pd

from constants import TARGET_PATIENT_FILE, TEMP_PATH
from nbconvert.preprocessors import ExecutePreprocessor


def getTargetPatientIcu() :
    PATIENT_PATH = TEMP_PATH / TARGET_PATIENT_FILE

    if not PATIENT_PATH.exists():
        nb = nbformat.read("./patients.ipynb", as_version=4)
        ep = ExecutePreprocessor(timeout=None, kernel_name="python3")
        
        resultNb, _ = ep.preprocess(nb)
        pass
        
    df = pd.read_csv(PATIENT_PATH)
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df
