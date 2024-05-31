from utils.extract_mesurements import extractLabEventMesures
from mimic_sql import (
    blood_differential,
    chemistry,
    complete_blood_count,
    first_day_lab_first_mesure,
)
from utils.reduce_mesurements import reduceByHadmId
from variables.demographics import getAge, getGender
from variables.lab_test.egfr import calculate_eGFR_df


def extractFirstDayLab():
    """Private

    Returns:
        _type_: _description_
    """
    return first_day_lab_first_mesure.runSql()


def extractSingularVariable(df, variableInDf, variableName):
    df = df[["stay_id", "charttime", variableInDf]]
    df = df.dropna(subset=[variableInDf])

    df = df.rename(columns={"charttime": "time", variableInDf: variableName})

    return df.sort_values(["stay_id", "time"])


# def getFirstMesureById(id: int, valueName: str = "valuenum"):
#     """Private. Extract a labevent by id. Reduce it by target patients.
#     Match the first value from -6h -> +24h of admittime.
#     Set value column name to mesureName.

#     Args:
#         id (int): labevent's item_id
#         mesureName (str): name of returned column

#     Returns:
#         pd.DataFrame: Dataframe consists of 2 columns: stay_id, mesureName
#     """

#     # def nonNullFirst(group: DataFrame):
#     #     groupNonNull = group.dropna(subset=["valuenum"])  # non-null
#     #     groupNonNull = groupNonNull.sort_values("charttime")
#     #     if groupNonNull.empty:
#     #         return nan
#     #     else:
#     #         return groupNonNull.iloc[0]["valuenum"]  # first row

#     df = extractLabEventMesures(id, "labevent-" + str(id) + ".csv")
#     dfReduced = reduceByHadmId(df)

#     # mesure may be performed multiple time, so get max of all
#     dfMaxPerSpeciment = dfReduced
#     dfMaxPerSpeciment["valuenum"] = \
#         dfReduced\
#         .groupby("specimen_id")["valuenum"]\
#         .transform("max")
#     dfMaxPerSpeciment.drop_duplicates("specimen_id", inplace=True)

#     result = (
#         dfMaxPerSpeciment
#         .groupby("stay_id")
#         .apply(nonNullFirst)
#         .reset_index(name=valueName)
#     )

#     return result


def getWbc():
    dfCBCReduced = reduceByHadmId(complete_blood_count.runSql())

    return extractSingularVariable(dfCBCReduced, "wbc", "wbc")


def getLymphocyte():
    dfBDReduced = reduceByHadmId(blood_differential.runSql())

    return extractSingularVariable(dfBDReduced, "lymphocytes_abs", "lymphocyte")


def getHb():
    dfCBCReduced = reduceByHadmId(complete_blood_count.runSql())

    return extractSingularVariable(dfCBCReduced, "hemoglobin", "hb")


def getPlt():
    dfCBCReduced = reduceByHadmId(complete_blood_count.runSql())

    return extractSingularVariable(dfCBCReduced, "platelet", "plt")


def getPO2():
    df = extractLabEventMesures(50821, "labevent-po2.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "po2", "charttime": "time"}
    )


def getPCO2():
    df = extractLabEventMesures(50818, "labevent-pco2.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "pco2", "charttime": "time"}
    )


def get_pH():
    df = extractLabEventMesures(50820, "labevent-ph.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "ph", "charttime": "time"}
    )


def getAG():
    """anion gap

    Returns: "ag"
    """

    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "aniongap", "ag")


def getBicarbonate():
    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "bicarbonate", "bicarbonate")


def getBun():
    """blood urea nitrogen

    Returns: "bun"
    """

    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "bun", "bun")


def getCalcium():
    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "calcium", "calcium")


def getScr():
    """serum creatinine

    Returns: "scr"
    """

    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "creatinine", "scr")


def getBg():
    """blood glucose

    Returns: "bg"
    """
    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "glucose", "bg")


def getPhosphate():
    df = extractLabEventMesures(50970, "labevent-phosphate.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "phosphate", "charttime": "time"}
    )


def getAlbumin():
    df = reduceByHadmId(chemistry.runSql())

    return extractSingularVariable(df, "albumin", "albumin")


def get_eGFR():
    dfCreat = getScr()
    dfAge = getAge()
    dfGender = getGender()

    dfMerged = dfCreat.merge(dfAge, "inner", "stay_id").merge(
        dfGender, "inner", "stay_id"
    )

    return calculate_eGFR_df(dfMerged)


def getHbA1C():
    df = extractLabEventMesures(50852, "labevent-hba1c.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "hba1c", "charttime": "time"}
    )


def getCrp():
    df = extractLabEventMesures(50889, "labevent-crp.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "crp", "charttime": "time"}
    )


def getUrineKetone():
    df = extractLabEventMesures(51484, "labevent-urine-ketone.csv")
    return reduceByHadmId(df)[["stay_id", "valuenum", "charttime"]].rename(
        columns={"valuenum": "urine-ketone", "charttime": "time"}
    )


############ Other measures found not in original paper ############
