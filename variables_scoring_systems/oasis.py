from middle_query import oasis


def extractOasis():
    # the query limit time already, but values are duplicated
    df = oasis.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()
