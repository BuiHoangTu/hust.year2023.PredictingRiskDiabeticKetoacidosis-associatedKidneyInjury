from middle_query import first_day_lab_first_mesure


def extractFirstDayLab():
    """Private

    Returns:
        _type_: _description_
    """
    df = first_day_lab_first_mesure.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()


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
    return None

def getPCO2():
    return None

def get_pH():
    return None

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
    return None

def getAlbumin():
    df = extractFirstDayLab()
    df["albumin"] = df["albumin_first"]

    return df[["stay_id", "albumin"]]

def get_eGFR():
    return None

def getHbA1C():
    return None

def getCrp():
    return None

def getUrineKetone():
    return None
