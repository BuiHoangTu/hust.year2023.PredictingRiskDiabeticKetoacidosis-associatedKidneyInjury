from notebook_wrappers.target_patients_wrapper import getTargetPatientIcd


def get():
    df1 = getDiabeticNephropathy()
    df2 = getDiabeticRetinopathy()
    df3 = getDiabeticPeripheralNeuropathy()

    dfRes = df1.merge(df2, "outer", "hadm_id").merge(df3, "outer", "hadm_id")

    dfRes["microangiopathy"] = dfRes[["dn", "dr", "dpn"]].any(axis=1)

    return (
        dfRes[["hadm_id", "microangiopathy"]]
        .groupby("hadm_id")["microangiopathy"]
        .any()
        .reset_index()
    )


def getDiabeticNephropathy():
    # synonym: diabetic kidney disease
    codes = [
        "E0822",  # Diabetes mellitus due to underlying condition with diabetic chronic kidney disease
        "E0922",  # Drug or chemical induced diabetes mellitus with diabetic chronic kidney disease
        "E1022",  # Type 1 diabetes mellitus with diabetic chronic kidney disease
        "E1122",  # Type 2 diabetes mellitus with diabetic chronic kidney disease
        "E1322",  # Other specified diabetes mellitus with diabetic chronic kidney disease
        "E0821",  # Diabetes mellitus due to underlying condition with diabetic nephropathy
        "E0921",  # Drug or chemical induced diabetes mellitus with diabetic nephropathy
        "E1021",  # Type 1 diabetes mellitus with diabetic nephropathy
        "E1121",  # Type 2 diabetes mellitus with diabetic nephropathy
        "E1321",  # Other specified diabetes mellitus with diabetic nephropathy
    ]

    dfPatIcd = getTargetPatientIcd()

    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]

    dfRes = dfRes.copy()
    dfRes["dn"] = True

    return dfRes[["hadm_id", "dn"]]


def getDiabeticRetinopathy():
    codes = [
        "36201",  # Background diabetic retinopathy
        "36202",  # Proliferative diabetic retinopathy
        "36203",  # Nonproliferative diabetic retinopathy NOS
        "36204",  # Mild nonproliferative diabetic retinopathy
        "36205",  # Moderate nonproliferative diabetic retinopathy
        "36206",  # Severe nonproliferative diabetic retinopathy
        "E0831",  # Diabetes mellitus due to underlying condition with unspecified diabetic retinopathy
        "E08311",  # Diabetes mellitus due to underlying condition with unspecified diabetic retinopathy with macular edema
        "E08319",  # Diabetes mellitus due to underlying condition with unspecified diabetic retinopathy without macular edema
        "E0832",  # Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy
        "E08321",  # Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy with macular edema
        "E083211",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy with macular edema, right eye"
        "E083212",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy with macular edema, left eye"
        "E083213",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E083219",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E08329",  # Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy without macular edema
        "E083291",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy without macular edema, right eye"
        "E083292",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy without macular edema, left eye"
        "E083293",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E083299",  # "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0833",  # Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy
        "E08331",  # Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy with macular edema
        "E083311",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy with macular edema, right eye"
        "E083312",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy with macular edema, left eye"
        "E083313",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E083319",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E08339",  # Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy without macular edema
        "E083391",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy without macular edema, right eye"
        "E083392",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy without macular edema, left eye"
        "E083393",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E083399",  # "Diabetes mellitus due to underlying condition with moderate nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0834",  # Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy
        "E08341",  # Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy with macular edema
        "E083411",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy with macular edema, right eye"
        "E083412",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy with macular edema, left eye"
        "E083413",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E083419",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E08349",  # Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy without macular edema
        "E083491",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy without macular edema, right eye"
        "E083492",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy without macular edema, left eye"
        "E083493",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E083499",  # "Diabetes mellitus due to underlying condition with severe nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0835",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy
        "E08351",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with macular edema
        "E083511",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with macular edema, right eye"
        "E083512",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with macular edema, left eye"
        "E083513",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with macular edema, bilateral"
        "E083519",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with macular edema, unspecified eye"
        "E08352",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment involving the macula
        "E083521",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye"
        "E083522",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment involving the macula, left eye"
        "E083523",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment involving the macula, bilateral"
        "E083529",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment involving the macula, unspecified eye"
        "E08353",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment not involving the macula
        "E083531",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, right eye"
        "E083532",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, left eye"
        "E083533",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, bilateral"
        "E083539",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, unspecified eye"
        "E08354",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment
        "E083541",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, right eye"
        "E083542",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, left eye"
        "E083543",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, bilateral"
        "E083549",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, unspecified eye"
        "E08355",  # Diabetes mellitus due to underlying condition with stable proliferative diabetic retinopathy
        "E083551",  # "Diabetes mellitus due to underlying condition with stable proliferative diabetic retinopathy, right eye"
        "E083552",  # "Diabetes mellitus due to underlying condition with stable proliferative diabetic retinopathy, left eye"
        "E083553",  # "Diabetes mellitus due to underlying condition with stable proliferative diabetic retinopathy, bilateral"
        "E083559",  # "Diabetes mellitus due to underlying condition with stable proliferative diabetic retinopathy, unspecified eye"
        "E08359",  # Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy without macular edema
        "E083591",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy without macular edema, right eye"
        "E083592",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy without macular edema, left eye"
        "E083593",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy without macular edema, bilateral"
        "E083599",  # "Diabetes mellitus due to underlying condition with proliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0931",  # Drug or chemical induced diabetes mellitus with unspecified diabetic retinopathy
        "E09311",  # Drug or chemical induced diabetes mellitus with unspecified diabetic retinopathy with macular edema
        "E09319",  # Drug or chemical induced diabetes mellitus with unspecified diabetic retinopathy without macular edema
        "E0932",  # Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy
        "E09321",  # Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema
        "E093211",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye"
        "E093212",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, left eye"
        "E093213",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E093219",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E09329",  # Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema
        "E093291",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, right eye"
        "E093292",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, left eye"
        "E093293",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E093299",  # "Drug or chemical induced diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0933",  # Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy
        "E09331",  # Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema
        "E093311",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, right eye"
        "E093312",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, left eye"
        "E093313",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E093319",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E09339",  # Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema
        "E093391",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, right eye"
        "E093392",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, left eye"
        "E093393",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E093399",  # "Drug or chemical induced diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0934",  # Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy
        "E09341",  # Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema
        "E093411",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, right eye"
        "E093412",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, left eye"
        "E093413",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E093419",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E09349",  # Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema
        "E093491",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, right eye"
        "E093492",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, left eye"
        "E093493",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E093499",  # "Drug or chemical induced diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E0935",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy
        "E09351",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with macular edema
        "E093511",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with macular edema, right eye"
        "E093512",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with macular edema, left eye"
        "E093513",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with macular edema, bilateral"
        "E093519",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with macular edema, unspecified eye"
        "E09352",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula
        "E093521",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye"
        "E093522",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, left eye"
        "E093523",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, bilateral"
        "E093529",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, unspecified eye"
        "E09353",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula
        "E093531",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, right eye"
        "E093532",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, left eye"
        "E093533",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, bilateral"
        "E093539",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, unspecified eye"
        "E09354",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment
        "E093541",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, right eye"
        "E093542",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, left eye"
        "E093543",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, bilateral"
        "E093549",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, unspecified eye"
        "E09355",  # Drug or chemical induced diabetes mellitus with stable proliferative diabetic retinopathy
        "E093551",  # "Drug or chemical induced diabetes mellitus with stable proliferative diabetic retinopathy, right eye"
        "E093552",  # "Drug or chemical induced diabetes mellitus with stable proliferative diabetic retinopathy, left eye"
        "E093553",  # "Drug or chemical induced diabetes mellitus with stable proliferative diabetic retinopathy, bilateral"
        "E093559",  # "Drug or chemical induced diabetes mellitus with stable proliferative diabetic retinopathy, unspecified eye"
        "E09359",  # Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy without macular edema
        "E093591",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy without macular edema, right eye"
        "E093592",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy without macular edema, left eye"
        "E093593",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy without macular edema, bilateral"
        "E093599",  # "Drug or chemical induced diabetes mellitus with proliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1031",  # Type 1 diabetes mellitus with unspecified diabetic retinopathy
        "E10311",  # Type 1 diabetes mellitus with unspecified diabetic retinopathy with macular edema
        "E10319",  # Type 1 diabetes mellitus with unspecified diabetic retinopathy without macular edema
        "E1032",  # Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy
        "E10321",  # Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema
        "E103211",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye"
        "E103212",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, left eye"
        "E103213",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E103219",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E10329",  # Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema
        "E103291",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, right eye"
        "E103292",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, left eye"
        "E103293",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E103299",  # "Type 1 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1033",  # Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy
        "E10331",  # Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema
        "E103311",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, right eye"
        "E103312",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, left eye"
        "E103313",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E103319",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E10339",  # Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema
        "E103391",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, right eye"
        "E103392",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, left eye"
        "E103393",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E103399",  # "Type 1 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1034",  # Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy
        "E10341",  # Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema
        "E103411",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, right eye"
        "E103412",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, left eye"
        "E103413",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E103419",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E10349",  # Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema
        "E103491",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, right eye"
        "E103492",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, left eye"
        "E103493",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E103499",  # "Type 1 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1035",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy
        "E10351",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy with macular edema
        "E103511",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with macular edema, right eye"
        "E103512",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with macular edema, left eye"
        "E103513",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with macular edema, bilateral"
        "E103519",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with macular edema, unspecified eye"
        "E10352",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula
        "E103521",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye"
        "E103522",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, left eye"
        "E103523",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, bilateral"
        "E103529",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, unspecified eye"
        "E10353",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula
        "E103531",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, right eye"
        "E103532",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, left eye"
        "E103533",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, bilateral"
        "E103539",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, unspecified eye"
        "E10354",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment
        "E103541",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, right eye"
        "E103542",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, left eye"
        "E103543",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, bilateral"
        "E103549",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, unspecified eye"
        "E10355",  # Type 1 diabetes mellitus with stable proliferative diabetic retinopathy
        "E103551",  # "Type 1 diabetes mellitus with stable proliferative diabetic retinopathy, right eye"
        "E103552",  # "Type 1 diabetes mellitus with stable proliferative diabetic retinopathy, left eye"
        "E103553",  # "Type 1 diabetes mellitus with stable proliferative diabetic retinopathy, bilateral"
        "E103559",  # "Type 1 diabetes mellitus with stable proliferative diabetic retinopathy, unspecified eye"
        "E10359",  # Type 1 diabetes mellitus with proliferative diabetic retinopathy without macular edema
        "E103591",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy without macular edema, right eye"
        "E103592",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy without macular edema, left eye"
        "E103593",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy without macular edema, bilateral"
        "E103599",  # "Type 1 diabetes mellitus with proliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1131",  # Type 2 diabetes mellitus with unspecified diabetic retinopathy
        "E11311",  # Type 2 diabetes mellitus with unspecified diabetic retinopathy with macular edema
        "E11319",  # Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema
        "E1132",  # Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy
        "E11321",  # Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema
        "E113211",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye"
        "E113212",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, left eye"
        "E113213",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E113219",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E11329",  # Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema
        "E113291",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, right eye"
        "E113292",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, left eye"
        "E113293",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E113299",  # "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1133",  # Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy
        "E11331",  # Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema
        "E113311",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, right eye"
        "E113312",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, left eye"
        "E113313",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E113319",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E11339",  # Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema
        "E113391",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, right eye"
        "E113392",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, left eye"
        "E113393",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E113399",  # "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1134",  # Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy
        "E11341",  # Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema
        "E113411",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, right eye"
        "E113412",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, left eye"
        "E113413",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E113419",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E11349",  # Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema
        "E113491",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, right eye"
        "E113492",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, left eye"
        "E113493",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E113499",  # "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1135",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy
        "E11351",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema
        "E113511",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema, right eye"
        "E113512",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema, left eye"
        "E113513",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema, bilateral"
        "E113519",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema, unspecified eye"
        "E11352",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula
        "E113521",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye"
        "E113522",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, left eye"
        "E113523",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, bilateral"
        "E113529",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, unspecified eye"
        "E11353",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula
        "E113531",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, right eye"
        "E113532",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, left eye"
        "E113533",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, bilateral"
        "E113539",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, unspecified eye"
        "E11354",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment
        "E113541",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, right eye"
        "E113542",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, left eye"
        "E113543",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, bilateral"
        "E113549",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, unspecified eye"
        "E11355",  # Type 2 diabetes mellitus with stable proliferative diabetic retinopathy
        "E113551",  # "Type 2 diabetes mellitus with stable proliferative diabetic retinopathy, right eye"
        "E113552",  # "Type 2 diabetes mellitus with stable proliferative diabetic retinopathy, left eye"
        "E113553",  # "Type 2 diabetes mellitus with stable proliferative diabetic retinopathy, bilateral"
        "E113559",  # "Type 2 diabetes mellitus with stable proliferative diabetic retinopathy, unspecified eye"
        "E11359",  # Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema
        "E113591",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema, right eye"
        "E113592",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema, left eye"
        "E113593",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema, bilateral"
        "E113599",  # "Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1331",  # Other specified diabetes mellitus with unspecified diabetic retinopathy
        "E13311",  # Other specified diabetes mellitus with unspecified diabetic retinopathy with macular edema
        "E13319",  # Other specified diabetes mellitus with unspecified diabetic retinopathy without macular edema
        "E1332",  # Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy
        "E13321",  # Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema
        "E133211",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye"
        "E133212",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, left eye"
        "E133213",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E133219",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E13329",  # Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema
        "E133291",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, right eye"
        "E133292",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, left eye"
        "E133293",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E133299",  # "Other specified diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1333",  # Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy
        "E13331",  # Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema
        "E133311",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, right eye"
        "E133312",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, left eye"
        "E133313",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E133319",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E13339",  # Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema
        "E133391",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, right eye"
        "E133392",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, left eye"
        "E133393",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E133399",  # "Other specified diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1334",  # Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy
        "E13341",  # Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema
        "E133411",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, right eye"
        "E133412",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, left eye"
        "E133413",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, bilateral"
        "E133419",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy with macular edema, unspecified eye"
        "E13349",  # Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema
        "E133491",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, right eye"
        "E133492",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, left eye"
        "E133493",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, bilateral"
        "E133499",  # "Other specified diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema, unspecified eye"
        "E1335",  # Other specified diabetes mellitus with proliferative diabetic retinopathy
        "E13351",  # Other specified diabetes mellitus with proliferative diabetic retinopathy with macular edema
        "E133511",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with macular edema, right eye"
        "E133512",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with macular edema, left eye"
        "E133513",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with macular edema, bilateral"
        "E133519",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with macular edema, unspecified eye"
        "E13352",  # Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula
        "E133521",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye"
        "E133522",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, left eye"
        "E133523",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, bilateral"
        "E133529",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment involving the macula, unspecified eye"
        "E13353",  # Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula
        "E133531",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, right eye"
        "E133532",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, left eye"
        "E133533",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, bilateral"
        "E133539",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with traction retinal detachment not involving the macula, unspecified eye"
        "E13354",  # Other specified diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment
        "E133541",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, right eye"
        "E133542",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, left eye"
        "E133543",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, bilateral"
        "E133549",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy with combined traction retinal detachment and rhegmatogenous retinal detachment, unspecified eye"
        "E13355",  # Other specified diabetes mellitus with stable proliferative diabetic retinopathy
        "E133551",  # "Other specified diabetes mellitus with stable proliferative diabetic retinopathy, right eye"
        "E133552",  # "Other specified diabetes mellitus with stable proliferative diabetic retinopathy, left eye"
        "E133553",  # "Other specified diabetes mellitus with stable proliferative diabetic retinopathy, bilateral"
        "E133559",  # "Other specified diabetes mellitus with stable proliferative diabetic retinopathy, unspecified eye"
        "E13359",  # Other specified diabetes mellitus with proliferative diabetic retinopathy without macular edema
        "E133591",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy without macular edema, right eye"
        "E133592",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy without macular edema, left eye"
        "E133593",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy without macular edema, bilateral"
        "E133599",  # "Other specified diabetes mellitus with proliferative diabetic retinopathy without macular edema, unspecified eye"
    ]

    dfPatIcd = getTargetPatientIcd()

    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]

    dfRes = dfRes.copy()
    dfRes["dr"] = True

    return dfRes[["hadm_id", "dr"]]


def getDiabeticPeripheralNeuropathy():
    # https://www.mmplusinc.com/kb-articles/coding-type-2-diabetes-mellitus-with-peripheral-neuropathy

    codes = [
        "3572",  # Polyneuropathy in diabetes
        "E0842",  # Diabetes mellitus due to underlying condition with diabetic polyneuropathy
        "E0942",  # Drug or chemical induced diabetes mellitus with neurological complications with diabetic polyneuropathy
        "E1042",  # Type 1 diabetes mellitus with diabetic polyneuropathy
        "E1142",  # Type 2 diabetes mellitus with diabetic polyneuropathy
        "E1342",  # Other specified diabetes mellitus with diabetic polyneuropathy
        "E0840",  # "Diabetes mellitus due to underlying condition with diabetic neuropathy, unspecified"
        "E0940",  # "Drug or chemical induced diabetes mellitus with neurological complications with diabetic neuropathy, unspecified"
        "E1040",  # "Type 1 diabetes mellitus with diabetic neuropathy, unspecified"
        "E1140",  # "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"
        "E1340",  # "Other specified diabetes mellitus with diabetic neuropathy, unspecified"
    ]

    dfPatIcd = getTargetPatientIcd()

    dfRes = dfPatIcd[dfPatIcd["icd_code"].isin(codes)]

    dfRes = dfRes.copy()
    dfRes["dpn"] = True

    return dfRes[["hadm_id", "dpn"]]
