from notebook_wrappers.target_patients_wrapper import getTargetPatientIcd


def get():
    df1 = getCoronaryHeartDisease()
    df2 = getCerebralAtherosclerosis()
    df3 = getPeripheralAtherosclerosis()

    dfRes = df1.merge(df2, "outer", "hadm_id").merge(df3, "outer", "hadm_id")

    dfRes["macroangiopathy"] = dfRes[["coronary", "ca", "pa"]].any(axis=1)

    return (
        dfRes[["hadm_id", "macroangiopathy"]]
        .groupby("hadm_id")["macroangiopathy"]
        .any()
        .reset_index()
    )

def getCoronaryHeartDisease():
    """Coronary heart disease corresponds to codes 
    410-414 in ICD-9 and 
    I20-I25 in ICD-10.
    """
    # https://www.chp.gov.hk/en/healthtopics/content/25/57.html

    dfDiagnosesIcd = getTargetPatientIcd()

    code9 = [str(i) for i in range(410, 414 + 1)]
    code10 = ["I" + str(i) for i in range(20, 25 + 1)]
    codePrefixes = code9 + code10

    dfCoronary = dfDiagnosesIcd[
        dfDiagnosesIcd["icd_code"].str.startswith(tuple(codePrefixes))
    ]
    
    dfCoronary = dfCoronary.copy()
    dfCoronary["coronary"] = True

    return dfCoronary[["hadm_id", "coronary"]]

def getCerebralAtherosclerosis():
    dfDiagnosesIcd = getTargetPatientIcd()
    dfCA = dfDiagnosesIcd[dfDiagnosesIcd["icd_code"].isin(["I672", "4370"])]

    dfCA = dfCA.copy()
    dfCA["ca"] = True


    return dfCA[["hadm_id", "ca"]]

def getPeripheralAtherosclerosis():
    # https://www.fortherecordmag.com/archives/101110p28.shtml
    code9 = [
        4439,
        25070,
        25071,
        25072,
        25073,
        44381,
        44020,
        44021,
        44022,
        44023,
        44024,
        44029,
    ]
    # https://www.outsourcestrategies.com/blog/code-peripheral-artery-disease-pad/
    code10Prefix = ["I700", "I701", "I702", "I73"]

    dfDiagnosesIcd = getTargetPatientIcd()

    code9Filter = dfDiagnosesIcd["icd_code"].isin(code9)
    code10Filter = dfDiagnosesIcd["icd_code"].str.startswith(tuple(code10Prefix))

    dfCA = dfDiagnosesIcd[code9Filter | code10Filter]

    dfCA = dfCA.copy()
    dfCA["pa"] = True

    return dfCA[["hadm_id", "pa"]]
