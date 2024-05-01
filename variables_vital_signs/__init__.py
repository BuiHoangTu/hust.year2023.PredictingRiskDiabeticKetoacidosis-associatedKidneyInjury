from middle_query import vitalsign


def getHeartRate():
    """

    Returns:
        DataFrame: ["stay_id", "hr"]
    """

    df = vitalsign.runSql()
    df["hr"] = df["heart_rate_first"]
    return df[["stay_id", "hr"]]


def getRespiratoryRate():
    """

    Returns:
        DataFrame: ["stay_id", "rr"]
    """

    df = vitalsign.runSql()
    df["rr"] = df["resp_rate_first"]
    return df[["stay_id", "rr"]]

def getSystolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "sbp"]
    """
    
    df = vitalsign.runSql()
    df["sbp"] = df["sbp_first"]
    return df[["stay_id", "sbp"]]

def getDiastolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "dbp"]
    """
    
    df = vitalsign.runSql()
    df["dbp"] = df["dbp_first"]
    return df[["stay_id", "dbp"]]