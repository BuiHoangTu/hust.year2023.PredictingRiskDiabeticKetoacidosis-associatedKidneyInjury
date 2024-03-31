import pandas as pd
from constants import TARGET_PATIENT_FILE, TEMP_PATH
from extract_mesurements import extractChartEventMesures, extractOutputEvents


def markAkdUrineOutput():
    """read urine output mesures and determine in which mesure patients are possitive


    Returns:
        pd.DataFrame: creatinine mesure 24h within icu admission,
        added "aki_stage_ou" indicating akd stage of patients by the mesure
    """

    INVERT_URINE_ID = 227488

    OUTPUT_EVENT_URINE_IDs = [
        226559,
        226560,
        226561,
        226584,
        226563,
        226564,
        226565,
        226567,
        226557,
        226558,
        227488,
        227489,
    ]

    URINE_OUTPUT_FILE_NAME = "urine_output.csv"
    usingColumns = ["stay_id", "charttime", "itemid"]
    dfUrine = extractOutputEvents(
        OUTPUT_EVENT_URINE_IDs, URINE_OUTPUT_FILE_NAME
    )
    dfUrine = dfUrine[usingColumns]
    dfUrine["charttime"] = pd.to_datetime(dfUrine["charttime"])


    dfUrine.loc[(dfUrine["itemid"] == INVERT_URINE_ID) & (dfUrine["valuenum"] > 0), "valuenum"] *= 1

    dfSumUrine = dfUrine.groupby(["stay_id, charttime"]).sum()["valuenum"].reset_index()

    del dfUrine
    ######### urine output rate #########
    # icu_stay filtered
    dfIcuPatients = pd.read_csv(
        TEMP_PATH / TARGET_PATIENT_FILE,
        usecols=["stay_id", "intime", "outtime", "subject_id"],
        parse_dates=["intime", "outtime"],
    )

    EVENT_ID = 220045
    CHART_EVENT_FILE_220045 = "chartevent" + str(EVENT_ID)

    dfChartEvent220045 = extractChartEventMesures(EVENT_ID, CHART_EVENT_FILE_220045)

    dfChartEvent220045["charttime"] = pd.to_datetime(dfChartEvent220045["charttime"])

    dfTm = pd.merge(dfIcuPatients, dfChartEvent220045, on="stay_id", how="inner")

    dfTm = dfTm[
        (dfTm["charttime"] > (dfTm["intime"] - pd.Timedelta(days=31))) &
        (dfTm["charttime"] < (dfTm["outime"] + pd.Timedelta(days=31)))
    ]

    dfTm = dfTm.groupby("stay_id").agg(
        intime_hr=("charttime", "min"), 
        outtime_hr=("charttime", "max"),
    ).reset_index()

    dfUoTm = pd.merge(dfTm, dfSumUrine, on="stay_id", how="inner")

    dfUoTm["tm_since_last_uo"] = dfUoTm\
        .sort_values(["stay_id", "charttime"])\
        .groupby("stay_id")["charttime"].diff()\
        .fillna(dfUoTm["charttime"] - dfUoTm["intime_hr"])

    dfUrStg = pd.merge(dfUoTm, dfUoTm, on="stay_id", suffixes=("", "_sum"))

    dfUrStg = dfUrStg[
        (dfUrStg["charttime"] >= dfUrStg["charttime_sum"]) &
        (dfUrStg["charttime"] <= dfUrStg["charttime_sum"] + pd.Timedelta(hours=23))
    ]
    
    def calculate_uo(df, hours):
        return df['urineoutput_sum'][(df['charttime'] - df['charttime_sum']).dt.total_seconds() / 3600 <= hours].sum()

    def calculate_uo_tm(df, hours):
        return df['tm_since_last_uo_sum'][(df['charttime'] - df['charttime_sum']).dt.total_seconds() / 3600 <= hours].sum() / 60.0

    uo_6hr = dfUrStg.groupby(['stay_id', 'charttime']).apply(lambda x: calculate_uo(x, 6))
    uo_tm_6hr = dfUrStg.groupby(['stay_id', 'charttime']).apply(lambda x: calculate_uo_tm(x, 6))

    uo_12hr = dfUrStg.groupby(['stay_id', 'charttime']).apply(lambda x: calculate_uo(x, 12))
    uo_tm_12hr = dfUrStg.groupby(['stay_id', 'charttime']).apply(lambda x: calculate_uo_tm(x, 12))

    uo_24hr = dfUrStg.groupby(['stay_id', 'charttime'])['urineoutput_sum'].sum()
    uo_tm_24hr = dfUrStg.groupby(['stay_id', 'charttime'])['tm_since_last_uo_sum'].sum() / 60.0

    result_df = pd.DataFrame({  # noqa: F841
        'uo': uo_24hr,
        'urineoutput_6hr': uo_6hr,
        'uo_tm_6hr': uo_tm_6hr,
        'urineoutput_12hr': uo_12hr,
        'uo_tm_12hr': uo_tm_12hr,
        'urineoutput_24hr': uo_24hr,
        'uo_tm_24hr': uo_tm_24hr
    }).reset_index()
    ######### paper_query method #########

    # merge df
    dfCreatMesure = dfIcuPatients.merge(
        dfSumUrine,
        on="subject_id",
        how="left",
        suffixes=("_patient", "_creat"),
    )

    dfCreatMesure = dfCreatMesure[
        (dfCreatMesure["charttime"] >= (dfCreatMesure["intime"] - pd.Timedelta(days=7)))
        & (
            dfCreatMesure["charttime"]
            <= (dfCreatMesure["intime"] + pd.Timedelta(days=7))
        )
    ]

    # compute min value by slide window of n hours
    # shift this min value down
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

    #################################
    return dfCreatMesure


if __name__ == "__main__":
    df = markAkdUrineOutput()
    df.to_csv(TEMP_PATH / "akd_creatinine.csv")
    pass
