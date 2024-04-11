import pandas as pd
from akd_stage.crrt import extractCrrt
from constants import TARGET_PATIENT_FILE, TEMP_PATH


def extractUseCrrt():
    dfCrrt = extractCrrt()

    dfCrrt = dfCrrt.drop_duplicates("stay_id")

    dfCrrt["use_crrt"] = True

    dfTargetPatient = pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE)
    dfTargetPatient = dfTargetPatient[["stay_id", "intime"]]
    dfTargetPatient["intime"] = pd.to_datetime(dfTargetPatient["intime"])

    dfMerged = pd.merge(dfCrrt, dfTargetPatient, "inner", "stay_id")
    dfMerged = dfMerged[
        (dfMerged["charttime"] > (dfMerged["intime"] - pd.Timedelta(hours=6)))
        & (dfMerged["charttime"] < (dfMerged["intime"] + pd.Timedelta(hours=24)))
    ]

    return dfMerged[["stay_id", "use_crrt"]]
