from mimic_sql import vitalsign


def getHeartRate():
    """

    Returns:
        DataFrame: ["stay_id", "hr"]
    """

    df = vitalsign.runSql()
    df["hr"] = df["heart_rate"]
    return df[["stay_id", "hr", "charttime"]].rename(columns={"charttime": "time"})


def getRespiratoryRate():
    """

    Returns:
        DataFrame: ["stay_id", "rr"]
    """

    df = vitalsign.runSql()
    df["rr"] = df["resp_rate"]
    return df[["stay_id", "rr", "charttime"]].rename(columns={"charttime": "time"})

def getSystolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "sbp"]
    """

    df = vitalsign.runSql()
    df["sbp"] = df["sbp"]
    return df[["stay_id", "sbp", "charttime"]].rename(columns={"charttime": "time"})

def getDiastolicBloodPressure():
    """

    Returns:
        DataFrame: ["stay_id", "dbp"]
    """

    df = vitalsign.runSql()
    df["dbp"] = df["dbp"]
    return df[["stay_id", "dbp", "charttime"]].rename(columns={"charttime": "time"})
