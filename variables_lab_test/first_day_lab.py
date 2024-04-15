from middle_query import first_day_lab


def extractFirstDayLab():
    df = first_day_lab.runSql()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()
