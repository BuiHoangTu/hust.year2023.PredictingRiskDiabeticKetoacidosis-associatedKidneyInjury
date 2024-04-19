from middle_query import first_day_gcs, oasis


def getGcs():
    df = first_day_gcs.runSql()

    df["gcs"] = df["gcs_min"]
    df = df[["stay_id", "gcs"]]

    return df


def getOasis():
    # the query limit time already, but values are duplicated
    df = oasis.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()

    return df