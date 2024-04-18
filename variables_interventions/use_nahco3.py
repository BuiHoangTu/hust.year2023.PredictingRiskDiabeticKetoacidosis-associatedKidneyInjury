from extract_mesurements import extractInputEvents
from reduce_mesurements import reduceByStayId


def get():
    dfNahco3 = extractInputEvents([220995, 221211, 227533], "use_nahco3.csv")

    dfReduced = reduceByStayId(dfNahco3, "starttime", "endtime")

    dfReduced.drop_duplicates("stay_id", inplace=True)

    dfReduced["use_NaHCO3"] = True

    return dfReduced[["stay_id", "use_NaHCO3"]]
