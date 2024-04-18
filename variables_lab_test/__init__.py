from numpy import nan
from pandas import DataFrame
from extract_mesurements import extractLabEventMesures
from middle_query import first_day_lab_first_mesure
from reduce_mesurements import reduceByHadmId
from variables_demographics import getAge, getGender
from variables_lab_test.egfr import calculate_eGFR_df


def extractFirstDayLab():
    """Private

    Returns:
        _type_: _description_
    """
    return first_day_lab_first_mesure.runSql()


def getFirstMesureById(id: int, valueName: str = "valuenum"):
    """Private. Extract a labevent by id. Reduce it by target patients.
    Match the first value from -6h -> +24h of admittime.
    Set value column name to mesureName.

    Args:
        id (int): labevent's item_id
        mesureName (str): name of returned column

    Returns:
        pd.DataFrame: Dataframe consists of 2 columns: stay_id, mesureName
    """

    def nonNullFirst(group: DataFrame):
        groupNonNull = group.dropna(subset=["valuenum"])  # non-null
        groupNonNull = groupNonNull.sort_values("charttime")
        if (groupNonNull.empty):
            return nan
        else:
            return groupNonNull.iloc[0]["valuenum"] # first row

    df = extractLabEventMesures(id, "labevent-" + str(id) + ".csv")
    dfReduced = reduceByHadmId(df)

    # mesure may be performed multiple time, so get max of all
    dfMaxPerSpeciment = dfReduced
    dfMaxPerSpeciment["valuenum"] = \
        dfReduced\
        .groupby("specimen_id")["valuenum"]\
        .transform("max")
    dfMaxPerSpeciment.drop_duplicates("specimen_id", inplace=True)

    result = (
        dfMaxPerSpeciment
        .groupby("stay_id")
        .apply(nonNullFirst)
        .reset_index(name=valueName)
    )

    return result


def getWbc():
    df = extractFirstDayLab()
    df["wbc"] = df["wbc_first"]

    return df[["stay_id", "wbc"]]


def getLymphocyte():
    df = extractFirstDayLab()
    df["lymphocyte"] = df["abs_lymphocytes_first"]

    return df[["stay_id", "lymphocyte"]]


def getHb():
    df = extractFirstDayLab()
    df["hb"] = df["hemoglobin_first"]

    return df[["stay_id", "hb"]]


def getPlt():
    df = extractFirstDayLab()
    df["plt"] = df["platelets_first"]

    return df[["stay_id", "plt"]]


def getPO2():
    return getFirstMesureById(50821, "po2")


def getPCO2():
    return getFirstMesureById(50818, "pco2")


def get_pH():
    return getFirstMesureById(50820, "ph")


def getAG():
    """anion gap

    Returns: "ag"
    """

    df = extractFirstDayLab()
    df["ag"] = df["aniongap_first"]

    return df[["stay_id", "ag"]]


def getBicarbonate():
    df = extractFirstDayLab()
    df["bicarbonate"] = df["bicarbonate_first"]

    return df[["stay_id", "bicarbonate"]]


def getBun():
    """blood urea nitrogen

    Returns: "bun"
    """

    df = extractFirstDayLab()
    df["bun"] = df["bun_first"]

    return df[["stay_id", "bun"]]


def getCalcium():
    df = extractFirstDayLab()
    df["calcium"] = df["calcium_first"]

    return df[["stay_id", "calcium"]]


def getScr():
    """serum creatinine

    Returns: "scr"
    """

    df = extractFirstDayLab()
    df["scr"] = df["creatinine_first"]

    return df[["stay_id", "scr"]]


def getBg():
    """blood glucose

    Returns: "bg"
    """

    df = extractFirstDayLab()
    df["bg"] = df["glucose_first"]

    return df[["stay_id", "bg"]]


def getPhosphate():
    return getFirstMesureById(50970, "phosphate")


def getAlbumin():
    df = extractFirstDayLab()
    df["albumin"] = df["albumin_first"]

    return df[["stay_id", "albumin"]]


def get_eGFR():
    dfCreat = getScr()
    dfAge = getAge()
    dfGender = getGender()
    
    dfMerged = dfCreat\
        .merge(dfAge, "inner", "stay_id")\
            .merge(dfGender, "inner", "stay_id")
    
    return calculate_eGFR_df(dfMerged)


def getHbA1C():
    return getFirstMesureById(50852, "hba1c")


def getCrp():
    return getFirstMesureById(50889, "crp")


def getUrineKetone():
    return getFirstMesureById(51484, "urine-ketone")
