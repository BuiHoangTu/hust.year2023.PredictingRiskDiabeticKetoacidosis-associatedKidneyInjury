import numpy as np
from middle_query import charlson
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

# def getPreExistingCKD():
    