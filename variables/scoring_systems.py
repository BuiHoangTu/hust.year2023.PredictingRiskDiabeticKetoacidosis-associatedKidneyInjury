from mimic_sql import first_day_sofa, gcs, oasis, sapsii
from utils.reduce_mesurements import reduceByStayId


def getGcs():
    """Glasgow coma scale

    Returns:
        pandas.DataFrame: ["stay_id", "gcs", "time"]
    """

    df = gcs.runSql()

    df = reduceByStayId(df)

    df = df.rename(columns={"charttime": "time"})

    return df[["stay_id", "gcs", "gcs_unable", "time"]]


def getOasis():
    """Oxford acute severity of illness score

    The score is calculated on the first day of each ICU patients' stay.

    Returns:
        pandas.DataFrame: ["stay_id", "gcs"]
    """

    # the query limit time already, but values are duplicated
    df = oasis.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()

    return df[["stay_id", "oasis"]]


def getSofa():
    """Sequential Organ Failure Assessment.

    The score is calculated with 24h windows so one result only.

    Returns:
        pandas.DataFrame: ["stay_id", "sofa"]
    """
    return first_day_sofa.runSql()[["stay_id", "sofa"]]


def getSaps2():
    """simplified acute physiology score II.

    The score is calculated on the first day of each ICU patients' stay.

    Returns:
        pandas.DataFrame: ["stay_id", "saps2"]
    """

    df = sapsii.runSql()
    df["saps2"] = df["sapsii"]

    return df[["stay_id", "saps2"]]
