from mimic_sql import oasis


def getPreIcuLos():
    """Get time length patients stayed in hospital before Icu admission

    Returns:
        pandas.DataFrame: ["stay_id", "preiculos"]
    """
    # the query limit time already, but values are duplicated
    df = oasis.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()

    return df[["stay_id", "preiculos"]]
