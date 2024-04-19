from middle_query import first_day_gcs


def get():
    df = first_day_gcs.runSql()

    df["gcs"] = df["gcs_min"]
    df = df[["stay_id", "gcs"]]

    return df
