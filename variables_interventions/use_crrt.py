from akd_stage.crrt import extractCrrt


def extractUseCrrt():
    dfCrrt = extractCrrt()

    dfCrrt = dfCrrt[["stay_id"]].drop_duplicates("stay_id")

    dfCrrt["use_crrt"] = True

    return dfCrrt
