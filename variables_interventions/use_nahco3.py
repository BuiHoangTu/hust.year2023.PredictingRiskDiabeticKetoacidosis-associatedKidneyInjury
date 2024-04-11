from extract_mesurements import extractInputEvents
from reduce_mesurements import reduceByCharttime


def extractUseOfNaHCO3():
    dfNahco3 = extractInputEvents([220995, 221211, 227533], "use_nahco3.csv")

    dfReduced = reduceByCharttime(dfNahco3, "starttime", "endtime")

    dfReduced.drop_duplicates("stay_id", inplace=True)

    dfReduced["use_NaHCO3"] = True

    return dfReduced[["stay_id", "use_NaHCO3"]]
