from extract_mesurements import extractLabEventMesures
from reduce_mesurements import reduceByHadmId


def get():
    df = extractLabEventMesures(50820, "ph.csv")
    dfReduced = reduceByHadmId(df)

    result = (
        dfReduced
        .groupby("stay_id")
        .apply(
            lambda group: group.dropna(subset=["valuenum"])
            .sort_values("charttime")
            .iloc[0]["valuenum"]
        )
        .reset_index(name="ph")
    )

    return result
