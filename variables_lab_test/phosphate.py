from extract_mesurements import extractLabEventMesures
from reduce_mesurements import reduceByHadmId


def get():
    df = extractLabEventMesures(50970, "labevent-phosphate.csv")
    dfReduced = reduceByHadmId(df)

    dfMaxPerSpeciment = dfReduced
    dfMaxPerSpeciment["valuenum"] = dfReduced.groupby("specimen_id")["valuenum"].transform("max")
    dfMaxPerSpeciment.drop_duplicates("specimen_id", inplace=True)

    result = (
        dfMaxPerSpeciment
        .groupby("stay_id")
        .apply(
            lambda group: group.dropna(subset=["valuenum"])
            .sort_values("charttime")
            .iloc[0]["valuenum"]
        )
        .reset_index(name="phosphate")
    )

    return result
