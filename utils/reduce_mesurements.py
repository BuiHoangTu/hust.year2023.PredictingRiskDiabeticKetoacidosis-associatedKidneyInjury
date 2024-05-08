import pandas as pd

from constants import TARGET_PATIENT_FILE, TEMP_PATH


def reduceByStayId(df: pd.DataFrame, starttimeCol="charttime", endtimeCol="charttime"):
    """Reduce results of df. Get only data from 6h before to 24h after intime

    Args:
        df (pd.DataFrame): must include "charttime"

    Returns:
        pd.DataFrame: data from 6h before to 24h after intime
    """

    df[starttimeCol] = pd.to_datetime(df[starttimeCol])
    df[endtimeCol] = pd.to_datetime(df[endtimeCol])

    dfTargetPatient = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatient = dfTargetPatient[["stay_id", "intime"]]
    dfTargetPatient["intime"] = pd.to_datetime(dfTargetPatient["intime"])

    dfMerged = pd.merge(df, dfTargetPatient, "inner", "stay_id")
    dfMerged = dfMerged[
        (dfMerged[starttimeCol] > (dfMerged["intime"] - pd.Timedelta(hours=6)))
        & (dfMerged[endtimeCol] < (dfMerged["intime"] + pd.Timedelta(hours=24)))
    ]

    return dfMerged

def reduceByHadmId(df: pd.DataFrame, starttimeCol="charttime", endtimeCol="charttime"):
    df[starttimeCol] = pd.to_datetime(df[starttimeCol])
    df[endtimeCol] = pd.to_datetime(df[endtimeCol])

    dfTargetPatient = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatient = dfTargetPatient[["hadm_id", "stay_id", "intime"]]
    dfTargetPatient["intime"] = pd.to_datetime(dfTargetPatient["intime"])

    dfMerged = pd.merge(df, dfTargetPatient, "inner", "hadm_id")
    dfMerged = dfMerged[
        (dfMerged[starttimeCol] > (dfMerged["intime"] - pd.Timedelta(hours=6)))
        & (dfMerged[endtimeCol] < (dfMerged["intime"] + pd.Timedelta(hours=24)))
    ]

    return dfMerged