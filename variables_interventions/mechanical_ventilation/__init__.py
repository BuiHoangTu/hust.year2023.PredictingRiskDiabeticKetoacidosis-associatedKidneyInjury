import pandas as pd
from constants import TARGET_PATIENT_FILE, TEMP_PATH
from variables_interventions.mechanical_ventilation.ventilation import (
    extractVentilation,
)


def extractMechVent():
    dfMV = extractVentilation()
    dfMV["mechanical_ventilation"] = dfMV["ventilation_status"].isin(
        [
            "Tracheostomy",
            "InvasiveVent",
        ]
    )
    dfMV["starttime"] = pd.to_datetime(dfMV["starttime"])
    dfMV["endtime"] = pd.to_datetime(dfMV["endtime"])

    dfTargetPatient = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatient = dfTargetPatient[["stay_id", "intime"]]
    dfTargetPatient["intime"] = pd.to_datetime(dfTargetPatient["intime"])

    dfMerged = pd.merge(dfMV, dfTargetPatient, "inner", "stay_id")
    dfMerged = dfMerged[
        (dfMerged["starttime"] > (dfMerged["intime"] - pd.Timedelta(hours=6)))
        & (dfMerged["endtime"] < (dfMerged["intime"] + pd.Timedelta(hours=24)))
    ]

    dfMerged = dfMerged[["stay_id", "mechanical_ventilation"]]
    dfMerged = dfMerged[dfMerged["mechanical_ventilation"]]

    return dfMerged.drop_duplicates("stay_id")
