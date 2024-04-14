import pandas as pd

from constants import TARGET_PATIENT_FILE, TEMP_PATH


def extractTargetPatients() :
    df = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    
    return df
