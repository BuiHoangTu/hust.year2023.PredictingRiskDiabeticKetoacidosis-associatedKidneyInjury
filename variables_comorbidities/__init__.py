import numpy as np
from middle_query import charlson
from patients import getTargetPatientIcd
from variables_comorbidities import history_of_ACI, history_of_AMI


def getHistoryACI():
    """history of Acute cerebral infarction

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    return history_of_ACI.get()


def getHistoryAMI():
    """history of Acute myocardial infarction

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    return history_of_AMI.get()


def getCHF():
    """history of Congestive heart failure

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    df = charlson.runSql()
    df["congestive_heart_failure"] = df["congestive_heart_failure"].astype(bool)

    return df[["hadm_id", "congestive_heart_failure"]]


def getLiverDisease():
    """history of Liver disease. SEVERE - MILD - NONE

    Returns:
        pandas.DataFrame: ["hadm_id", "liver_disease"]
    """

    df = charlson.runSql()

    df["mild_liver_disease"] = df["mild_liver_disease"].astype(bool)
    df["severe_liver_disease"] = df["severe_liver_disease"].astype(bool)

    df["liver_disease"] = np.where(
        df["severe_liver_disease"],
        "SEVERE",
        np.where(df["mild_liver_disease"], "MILD", "NONE"),
    )

    return df[["hadm_id", "liver_disease"]]


def getPreExistingCKD():
    """ Get worst CKD stage of patients.
    
    0: Unspecified; 1 -> 4: CKD Stage

    Returns:
        pandas.DataFrame: ["hadm_id", "ckd_stage"]
    """

    # icd code to ckd stage
    MAP_ICD_CKD_STAGE = {
        "5851": 1,
        "5852": 2,
        "5853": 3,
        "5854": 4,
        "5859": 0,  # Unspecified
        "N181": 1,
        "N182": 2,
        "N183": 3,
        "N184": 4,
        "N189": 0,  # Unspecified
    }

    df = getTargetPatientIcd()

    df["ckd_stage"] = df["icd_code"].map(MAP_ICD_CKD_STAGE)
    df.dropna(subset=["ckd_stage"], inplace=True)
    df["ckd_stage"] = df["ckd_stage"].astype(int)

    return df[["hadm_id", "ckd_stage"]].groupby("hadm_id").agg("max").reset_index()
