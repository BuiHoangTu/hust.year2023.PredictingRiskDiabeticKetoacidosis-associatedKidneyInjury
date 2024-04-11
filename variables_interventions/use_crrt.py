from akd_stage.crrt import extractCrrt
from reduce_mesurements import reduceByCharttime


def extractUseCrrt():
    dfCrrt = extractCrrt()
    dfCrrt = dfCrrt.drop_duplicates("stay_id")
    dfCrrt["use_crrt"] = True

    dfMerged = reduceByCharttime(dfCrrt)

    return dfMerged[["stay_id", "use_crrt"]]
