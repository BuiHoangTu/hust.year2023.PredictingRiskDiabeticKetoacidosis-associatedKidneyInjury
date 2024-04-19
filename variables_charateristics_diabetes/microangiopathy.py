from patients import getTargetPatientIcd


def get():
    df1 = getDiabeticNephropathy()
    df2 = getDiabeticRetinopathy()
    df3 = getDiabeticPeripheralNeuropathy()

    dfRes = df1.merge(df2, "outer", "hadm_id").merge(df3, "outer", "hadm_id")

    dfRes["microangiopathy"] = dfRes[["dn", "dr", "dpn"]].any(axis=1)

    return dfRes[["hadm_id", "macroangiopathy"]]


def getDiabeticNephropathy():
    # There is no icd9 code for this specific item
    codes = [
        "E0821",
        "E0921",
        "E1021",
        "E1121",
        "E1321",
    ]
    
    dfPatIcd = getTargetPatientIcd()
    
    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]
    
    dfRes["dn"] = True

    return dfRes[["hadm_id", "dn"]]

def getDiabeticRetinopathy():
    codes = [
        "36201",
        "36202",
        "36203",
        "36204",
        "36205",
        "36206",
        "E0831",
        "E08311",
        "E08319",
        "E0832",
        "E08321",
        "E083211",
        "E083212",
        "E083213",
        "E083219",
        "E08329",
        "E083291",
        "E083292",
        "E083293",
        "E083299",
        "E0833",
        "E08331",
        "E083311",
        "E083312",
        "E083313",
        "E083319",
        "E08339",
        "E083391",
        "E083392",
        "E083393",
        "E083399",
        "E0834",
        "E08341",
        "E083411",
        "E083412",
        "E083413",
        "E083419",
        "E08349",
        "E083491",
        "E083492",
        "E083493",
        "E083499",
        "E0835",
        "E08351",
        "E083511",
        "E083512",
        "E083513",
        "E083519",
        "E08352",
        "E083521",
        "E083522",
        "E083523",
        "E083529",
        "E08353",
        "E083531",
        "E083532",
        "E083533",
        "E083539",
        "E08354",
        "E083541",
        "E083542",
        "E083543",
        "E083549",
        "E08355",
        "E083551",
        "E083552",
        "E083553",
        "E083559",
        "E08359",
        "E083591",
        "E083592",
        "E083593",
        "E083599",
        "E0931",
        "E09311",
        "E09319",
        "E0932",
        "E09321",
        "E093211",
        "E093212",
        "E093213",
        "E093219",
        "E09329",
        "E093291",
        "E093292",
        "E093293",
        "E093299",
        "E0933",
        "E09331",
        "E093311",
        "E093312",
        "E093313",
        "E093319",
        "E09339",
        "E093391",
        "E093392",
        "E093393",
        "E093399",
        "E0934",
        "E09341",
        "E093411",
        "E093412",
        "E093413",
        "E093419",
        "E09349",
        "E093491",
        "E093492",
        "E093493",
        "E093499",
        "E0935",
        "E09351",
        "E093511",
        "E093512",
        "E093513",
        "E093519",
        "E09352",
        "E093521",
        "E093522",
        "E093523",
        "E093529",
        "E09353",
        "E093531",
        "E093532",
        "E093533",
        "E093539",
        "E09354",
        "E093541",
        "E093542",
        "E093543",
        "E093549",
        "E09355",
        "E093551",
        "E093552",
        "E093553",
        "E093559",
        "E09359",
        "E093591",
        "E093592",
        "E093593",
        "E093599",
        "E1031",
        "E10311",
        "E10319",
        "E1032",
        "E10321",
        "E103211",
        "E103212",
        "E103213",
        "E103219",
        "E10329",
        "E103291",
        "E103292",
        "E103293",
        "E103299",
        "E1033",
        "E10331",
        "E103311",
        "E103312",
        "E103313",
        "E103319",
        "E10339",
        "E103391",
        "E103392",
        "E103393",
        "E103399",
        "E1034",
        "E10341",
        "E103411",
        "E103412",
        "E103413",
        "E103419",
        "E10349",
        "E103491",
        "E103492",
        "E103493",
        "E103499",
        "E1035",
        "E10351",
        "E103511",
        "E103512",
        "E103513",
        "E103519",
        "E10352",
        "E103521",
        "E103522",
        "E103523",
        "E103529",
        "E10353",
        "E103531",
        "E103532",
        "E103533",
        "E103539",
        "E10354",
        "E103541",
        "E103542",
        "E103543",
        "E103549",
        "E10355",
        "E103551",
        "E103552",
        "E103553",
        "E103559",
        "E10359",
        "E103591",
        "E103592",
        "E103593",
        "E103599",
        "E1131",
        "E11311",
        "E11319",
        "E1132",
        "E11321",
        "E113211",
        "E113212",
        "E113213",
        "E113219",
        "E11329",
        "E113291",
        "E113292",
        "E113293",
        "E113299",
        "E1133",
        "E11331",
        "E113311",
        "E113312",
        "E113313",
        "E113319",
        "E11339",
        "E113391",
        "E113392",
        "E113393",
        "E113399",
        "E1134",
        "E11341",
        "E113411",
        "E113412",
        "E113413",
        "E113419",
        "E11349",
        "E113491",
        "E113492",
        "E113493",
        "E113499",
        "E1135",
        "E11351",
        "E113511",
        "E113512",
        "E113513",
        "E113519",
        "E11352",
        "E113521",
        "E113522",
        "E113523",
        "E113529",
        "E11353",
        "E113531",
        "E113532",
        "E113533",
        "E113539",
        "E11354",
        "E113541",
        "E113542",
        "E113543",
        "E113549",
        "E11355",
        "E113551",
        "E113552",
        "E113553",
        "E113559",
        "E11359",
        "E113591",
        "E113592",
        "E113593",
        "E113599",
        "E1331",
        "E13311",
        "E13319",
        "E1332",
        "E13321",
        "E133211",
        "E133212",
        "E133213",
        "E133219",
        "E13329",
        "E133291",
        "E133292",
        "E133293",
        "E133299",
        "E1333",
        "E13331",
        "E133311",
        "E133312",
        "E133313",
        "E133319",
        "E13339",
        "E133391",
        "E133392",
        "E133393",
        "E133399",
        "E1334",
        "E13341",
        "E133411",
        "E133412",
        "E133413",
        "E133419",
        "E13349",
        "E133491",
        "E133492",
        "E133493",
        "E133499",
        "E1335",
        "E13351",
        "E133511",
        "E133512",
        "E133513",
        "E133519",
        "E13352",
        "E133521",
        "E133522",
        "E133523",
        "E133529",
        "E13353",
        "E133531",
        "E133532",
        "E133533",
        "E133539",
        "E13354",
        "E133541",
        "E133542",
        "E133543",
        "E133549",
        "E13355",
        "E133551",
        "E133552",
        "E133553",
        "E133559",
        "E13359",
        "E133591",
        "E133592",
        "E133593",
        "E133599",
    ]

    dfPatIcd = getTargetPatientIcd()

    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]

    dfRes["dr"] = True

    return dfRes[["hadm_id", "dr"]]

def getDiabeticPeripheralNeuropathy():
    codes = [
        "E0840",
        "E0940",
        "E1040",
        "E1140",
        "E1340",
    ]

    dfPatIcd = getTargetPatientIcd()

    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]

    dfRes["dpn"] = True

    return dfRes[["hadm_id", "dpn"]]