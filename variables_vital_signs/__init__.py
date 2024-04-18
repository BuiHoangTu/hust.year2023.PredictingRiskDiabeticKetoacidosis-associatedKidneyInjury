from middle_query import first_day_vitalsign_first_mesure


def getHeartRate():
    """

    Returns:
        DataFrame: ["stay_id", "hr"]
    """

    df = first_day_vitalsign_first_mesure.runSql()
    df["hr"] = df["heart_rate_first"]
    return df[["stay_id", "hr"]]


def getRespiratoryRate():
    """

    Returns:
        DataFrame: ["stay_id", "rr"]
    """

    df = first_day_vitalsign_first_mesure.runSql()
    df["rr"] = df["resp_rate_first"]
    return df[["stay_id", "rr"]]

def getSystolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "sbp"]
    """
    
    df = first_day_vitalsign_first_mesure.runSql()
    df["sbp"] = df["sbp_first"]
    return df[["stay_id", "sbp"]]

def getDiastolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "dbp"]
    """
    
    df = first_day_vitalsign_first_mesure.runSql()
    df["dbp"] = df["dbp_first"]
    return df[["stay_id", "dbp"]]