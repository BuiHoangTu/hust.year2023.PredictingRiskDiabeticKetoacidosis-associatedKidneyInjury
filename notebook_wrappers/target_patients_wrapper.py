import pandas as pd

from constants import MIMIC_PATH, TARGET_PATIENT_FILE, TEMP_PATH

from notebook_wrapper import NotebookWrapper


def getNotebookOutput():
    """Private, get all output of the associated nb

    Raises:
        IOError: If IOError happend during nb running or nb took too much time saving it output to file

    Returns:
        Dataframe: data
    """

    PATIENT_PATH = TEMP_PATH / TARGET_PATIENT_FILE

    if PATIENT_PATH.exists():
        return pd.read_csv(PATIENT_PATH, parse_dates=["intime", "outtime"])
    else:
        dfTargetPatients: pd.DataFrame = NotebookWrapper("target_patients.ipynb", None, "dfTargetPatients").run()  # type: ignore
        dfTargetPatients.to_csv(PATIENT_PATH)
        return dfTargetPatients


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
