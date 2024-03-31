import pandas as pd
from constants import TEMP_PATH
from extract_mesurements import extractLabEventMesures

MESURES_CREAT_STAGE_FILE_NAME = "akd_creatinine.csv"
ICU_CREAT_STAGE_FILE_NAME = "icu_akd_stage_creat.csv"


def markAkdCreatinine():
    """read creatinine mesures and determine in which mesure patients are possitive


    Returns:
        pd.DataFrame: creatinine mesure 7d within icu admission,
        added "aki_stage_creat" indicating akd stage of patients by the mesure
    """

    if (TEMP_PATH / MESURES_CREAT_STAGE_FILE_NAME).exists():
        return pd.read_csv(TEMP_PATH / MESURES_CREAT_STAGE_FILE_NAME)

    LAB_CREAT_ID = 50912

    LAB_CREAT_FILE_NAME = "labevent_creatinine.csv"
    usingColumns = ["charttime", "valuenum", "subject_id"]
    dfCreatinine = extractLabEventMesures(LAB_CREAT_ID, LAB_CREAT_FILE_NAME)
    dfCreatinine = dfCreatinine[usingColumns]
    dfCreatinine["charttime"] = pd.to_datetime(dfCreatinine["charttime"])

    # icu_stay filtered
    dfTargetPatients = pd.read_csv(
        TEMP_PATH / "target_patients.csv",
        usecols=["stay_id", "intime", "outtime", "subject_id"],
        parse_dates=["intime", "outtime"],
    )

    # merge df
    dfCreatMesure = dfTargetPatients.merge(
        dfCreatinine,
        on="subject_id",
        how="left",
        suffixes=("_patient", "_creat"),
    )

    dfCreatMesure = dfCreatMesure[
        # previous mesures for baseline
        (dfCreatMesure["charttime"] >= (dfCreatMesure["intime"] - pd.Timedelta(days=7)))
        &
        # mesures for detecting akd
        (dfCreatMesure["charttime"] <= (dfCreatMesure["intime"] + pd.Timedelta(days=7)))
    ]

    # compute min value by slide window of n hours
    # shift this min value down to remove current
    def calculateMinNHours(dfGroup: pd.DataFrame, n):
        dfGroup.set_index("intime", inplace=True)

        minNHours = (
            dfGroup["valuenum"].rolling(str(n) + "H", closed="left").min().shift(1)
        )  # exclude current

        dfGroup["creat_low_" + str(n) + "h"] = minNHours
        return dfGroup.reset_index()

    dfCreatMesure: pd.DataFrame = (
        dfCreatMesure.groupby("stay_id")
        .apply(lambda group: calculateMinNHours(group, 48))
        .reset_index(drop=True)
    )

    dfCreatMesure: pd.DataFrame = (
        dfCreatMesure.groupby("stay_id")
        .apply(lambda group: calculateMinNHours(group, 7 * 24))
        .reset_index(drop=True)
    )

    dfCreatMesure["aki_stage_creat"] = 0

    for i, row in dfCreatMesure.iterrows():
        value = row["valuenum"]
        value48h = row["creat_low_48h"]
        value7d = row["creat_low_168h"]

        if value > (value7d * 3):
            dfCreatMesure.at[i, "aki_stage_creat"] = 3
        elif value > 4:
            if (value48h <= 3.7) or (value >= value7d * 1.5):
                dfCreatMesure.at[i, "aki_stage_creat"] = 3
        elif value >= value7d * 2:
            dfCreatMesure.at[i, "aki_stage_creat"] = 2
        elif (value >= value48h + 0.3) or (value >= value7d * 1.5):
            dfCreatMesure.at[i, "aki_stage_creat"] = 1
        pass
    dfCreatMesure.to_csv(TEMP_PATH / MESURES_CREAT_STAGE_FILE_NAME)

    return dfCreatMesure


def markIcuCreatinine():
    """Mark which icu_stay will got which akd_stage in 7 days by creatinine (get worst stage)"""

    usingColumns = ["stay_id", "aki_stage_creat"]

    dfCreatinineStage = markAkdCreatinine()
    pass

    dfCreatinineStage = dfCreatinineStage[usingColumns]

    dfIcuCreatinineStage = (
        dfCreatinineStage.groupby("stay_id")["aki_stage_creat"].max().reset_index()
    )

    return dfIcuCreatinineStage


if __name__ == "__main__":
    dfCreatinineStage = markIcuCreatinine()
    dfCreatinineStage.to_csv(TEMP_PATH / ICU_CREAT_STAGE_FILE_NAME)
    print("Icu creatinine stage is outputed to", ICU_CREAT_STAGE_FILE_NAME)
    pass
