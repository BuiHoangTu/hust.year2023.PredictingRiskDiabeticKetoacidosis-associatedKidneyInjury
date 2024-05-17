import pandas as pd
from constants import MIMIC_PATH
from utils.extract_mesurements import extractChartEventMesures
from mimic_sql import weight_durations
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu


def getAge():
    """Get approximate age of patients according to mimic-iv

    Returns:
        pd.DataFrame: consists of stay_id, age
    """

    # intime - anchor_year + anchor_age
    dfPatientICU = getTargetPatientIcu()  # intime
    dfPatient = pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv")

    dfMerged = pd.merge(dfPatientICU, dfPatient, "inner", on="subject_id")
    dfMerged["age"] = (
        pd.to_datetime(dfMerged["intime"]).dt.year
        - dfMerged["anchor_year"]
        + dfMerged["anchor_age"]
    )

    return dfMerged[["stay_id", "age"]]


def getGender():
    """Get patients's biological gender

    Returns:
        pd.DataFrame: consists of stay_id, gender(M,F)
    """

    dfPatientICU = getTargetPatientIcu()
    dfPatient = pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv")

    dfMerged = pd.merge(dfPatientICU, dfPatient, "inner", on="subject_id")

    return dfMerged[["stay_id", "gender"]]


def getEthnicity():
    """_summary_

    Returns:
        pd.DataFrame: consists of stay_id, race(str-capital)
    """
    dfPatientICU = getTargetPatientIcu()
    dfAdmission = pd.read_csv(MIMIC_PATH / "hosp/admissions.csv")

    dfMerged = pd.merge(dfPatientICU, dfAdmission, "inner", on="hadm_id")

    return dfMerged[["stay_id", "race"]]


def getHeight():
    df = extractChartEventMesures(
        [
            226707,  # inch
            226730,  # cm
        ],
        "chartted-all-height.csv",
    )

    def convertToCm(row):
        if row["itemid"] == 226707:  # inch
            height = row["valuenum"] * 2.54
        else:
            height = row["valuenum"]

        return height

    # convert to cm
    df["height"] = df.apply(convertToCm, axis=1)

    return df[["stay_id", "height"]]


def getWeight():
    dfWeight = weight_durations.runSql()

    dfWeight.dropna(subset="weight")

    dfWeight.rename(columns={"starttime": "time"}, inplace=True)

    dfWeight = dfWeight.sort_values(["stay_id", "time"])
    return dfWeight[["stay_id", "weight", "time"]]
