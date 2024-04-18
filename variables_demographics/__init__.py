import pandas as pd
from constants import MIMIC_PATH
from patients import getTargetPatientIcu


def getAge():
    """Get approximate age of patients according to mimic-iv

    Returns:
        pd.DataFrame: consists of stay_id, age
    """
     
    # intime - anchor_year + anchor_age
    dfPatientICU = getTargetPatientIcu() # intime 
    dfPatient = pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv")

    dfMerged = pd.merge(dfPatientICU, dfPatient, "inner", on="subject_id")
    dfMerged["age"] = pd.to_datetime(dfMerged["intime"]).dt.year - dfMerged["anchor_year"] + dfMerged["anchor_age"]

    return dfMerged[["stay_id", "age"]]

def getGender():
    """Get patients's biological gender

    Returns:
        pd.DataFrame: consists of stay_id, gender(M,F)
    """
    
    dfPatientICU = getTargetPatientIcu() # intime 
    dfPatient = pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv")

    dfMerged = pd.merge(dfPatientICU, dfPatient, "inner", on="subject_id")

    return dfMerged[["stay_id", "gender"]]
