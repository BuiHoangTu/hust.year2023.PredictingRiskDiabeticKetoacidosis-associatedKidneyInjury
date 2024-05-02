from time import sleep
import nbformat
import pandas as pd

from constants import MIMIC_PATH, TARGET_PATIENT_FILE, TEMP_PATH
from nbconvert.preprocessors import ExecutePreprocessor


def getNotebookOutput():
    """Private, get all output of the associated nb

    Raises:
        IOError: If IOError happend during nb running or nb took too much time saving it output to file

    Returns:
        Dataframe: data
    """
    
    PATIENT_PATH = TEMP_PATH / TARGET_PATIENT_FILE

    if not PATIENT_PATH.exists():
        nb = nbformat.read("./patients.ipynb", as_version=4)
        ep = ExecutePreprocessor(timeout=None, kernel_name="python3")

        resultNb, _ = ep.preprocess(nb)
        pass

    # wait for maximun 5*2 seconds
    for _ in range(5):
        if PATIENT_PATH.exists():
            break
        else:
            sleep(2)
    else:
        raise IOError(PATIENT_PATH.__str__() + " took too much time to write.")

    df = pd.read_csv(PATIENT_PATH)
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df


def getTargetPatientIcu():
    df = getNotebookOutput()

    return df[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ]
    ]


def getTargetPatientIcd():
    """Get Icd dianogses of target patients

    Returns:
        pd.Dataframe: equals to read_csv then filter patients
    """
    
    dfDiagnosesIcd = pd.read_csv(MIMIC_PATH / "hosp" / "diagnoses_icd.csv")
    dfDiagnosesIcd["icd_code"] = dfDiagnosesIcd["icd_code"].astype(str)
    patHadmIds = set(getTargetPatientIcu()["hadm_id"])
    dfDiagnosesIcd = dfDiagnosesIcd[dfDiagnosesIcd["hadm_id"].isin(patHadmIds)]

    return dfDiagnosesIcd


def getTargetPatientAdmission():
    dfAdmission = pd.read_csv(MIMIC_PATH / "hosp/admissions.csv")
    patHadmIds = set(getTargetPatientIcu()["hadm_id"])
    dfAdmission = dfAdmission[dfAdmission["hadm_id"].isin(patHadmIds)]
    
    return dfAdmission
