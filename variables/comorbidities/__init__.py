import numpy as np
from mimic_sql import charlson
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcd
from variables.comorbidities import history_of_ACI
from variables.comorbidities import history_of_AMI


def getHistoryACI():
    """history of Acute cerebral infarction

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    return history_of_ACI.get()


def getHistoryAMI():
    """history of Acute myocardial infarction

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    return history_of_AMI.get()


def getCHF():
    """Congestive heart failure

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    df = charlson.runSql()
    df["congestive_heart_failure"] = df["congestive_heart_failure"].astype(bool)

    return df[["hadm_id", "congestive_heart_failure"]]


def getLiverDisease():
    """Liver disease. SEVERE - MILD - NONE

    Returns:
        pandas.DataFrame: ["hadm_id", "liver_disease"]
    """

    df = charlson.runSql()

    df["mild_liver_disease"] = df["mild_liver_disease"].astype(bool)
    df["severe_liver_disease"] = df["severe_liver_disease"].astype(bool)

    df["liver_disease"] = np.where(
        df["severe_liver_disease"],
        "SEVERE",
        np.where(df["mild_liver_disease"], "MILD", "NONE"),
    )

    return df[["hadm_id", "liver_disease"]]


def getPreExistingCKD():
    """Get worst CKD stage of patients.

    0: Unspecified; 1 -> 4: CKD Stage

    Returns:
        pandas.DataFrame: ["hadm_id", "ckd_stage"]
    """

    # icd code to ckd stage
    MAP_ICD_CKD_STAGE = {
        "5851": 1,
        "5852": 2,
        "5853": 3,
        "5854": 4,
        "5859": 0,  # Unspecified
        "N181": 1,
        "N182": 2,
        "N183": 3,
        "N184": 4,
        "N189": 0,  # Unspecified
    }

    df = getTargetPatientIcd()

    df["ckd_stage"] = df["icd_code"].map(MAP_ICD_CKD_STAGE)
    df.dropna(subset=["ckd_stage"], inplace=True)
    df["ckd_stage"] = df["ckd_stage"].astype(int)

    return df[["hadm_id", "ckd_stage"]].groupby("hadm_id").agg("max").reset_index()


def getMalignantCancer():
    df = charlson.runSql()
    df["malignant_cancer"] = df["malignant_cancer"].astype(bool)

    return (
        df[["hadm_id", "malignant_cancer"]]
        .groupby("hadm_id")["malignant_cancer"]
        .any()
        .reset_index()
    )


def getHypertension():
    """True/False

    Returns:
        pandas.DataFrame: ["hadm_id", "hypertension"]
    """

    codes = [
        "3482",  # Benign intracranial hypertension
        "36504",  # Ocular hypertension
        "4010",  # Malignant essential hypertension
        "4011",  # Benign essential hypertension
        "4019",  # Unspecified essential hypertension
        "40501",  # Malignant renovascular hypertension
        "40509",  # Other malignant secondary hypertension
        "40511",  # Benign renovascular hypertension
        "40519",  # Other benign secondary hypertension
        "40591",  # Unspecified renovascular hypertension
        "40599",  # Other unspecified secondary hypertension
        "4160",  # Primary pulmonary hypertension
        "45930",  # Chronic venous hypertension without complications
        "45931",  # Chronic venous hypertension with ulcer
        "45932",  # Chronic venous hypertension with inflammation
        "45933",  # Chronic venous hypertension with ulcer and inflammation
        "45939",  # Chronic venous hypertension with other complication
        "5723",  # Portal hypertension
        "64200",  # "Benign essential hypertension complicating pregnancy, childbirth, and the puerperium, unspecified as to episode of care or not applicable"
        "64201",  # "Benign essential hypertension complicating pregnancy, childbirth, and the puerperium, delivered, with or without mention of antepartum condition"
        "64202",  # "Benign essential hypertension, complicating pregnancy, childbirth, and the puerperium, delivered, with mention of postpartum complication"
        "64203",  # "Benign essential hypertension complicating pregnancy, childbirth, and the puerperium, antepartum condition or complication"
        "64204",  # "Benign essential hypertension complicating pregnancy, childbirth, and the puerperium, postpartum condition or complication"
        "64210",  # "Hypertension secondary to renal disease, complicating pregnancy, childbirth, and the puerperium, unspecified as to episode of care or not applicable"
        "64211",  # "Hypertension secondary to renal disease, complicating pregnancy, childbirth, and the puerperium, delivered, with or without mention of antepartum condition"
        "64212",  # "Hypertension secondary to renal disease, complicating pregnancy, childbirth, and the puerperium, delivered, with mention of postpartum complication"
        "64213",  # "Hypertension secondary to renal disease, complicating pregnancy, childbirth, and the puerperium, antepartum condition or complication"
        "64214",  # "Hypertension secondary to renal disease, complicating pregnancy, childbirth, and the puerperium, postpartum condition or complication"
        "64220",  # "Other pre-existing hypertension complicating pregnancy, childbirth, and the puerperium, unspecified as to episode of care or not applicable"
        "64221",  # "Other pre-existing hypertension, complicating pregnancy, childbirth, and the puerperium, delivered, with or without mention of antepartum condition"
        "64222",  # "Other pre-existing hypertension, complicating pregnancy, childbirth, and the puerperium, delivered, with mention of postpartum complication"
        "64223",  # "Other pre-existing hypertension, complicating pregnancy, childbirth, and the puerperium, antepartum condition or complication"
        "64224",  # "Other pre-existing hypertension,complicating pregnancy, childbirth, and the puerperium, , postpartum condition or complication"
        "64270",  # "Pre-eclampsia or eclampsia superimposed on pre-existing hypertension, unspecified as to episode of care or not applicable"
        "64271",  # "Pre-eclampsia or eclampsia superimposed on pre-existing hypertension, delivered, with or without mention of antepartum condition"
        "64272",  # "Pre-eclampsia or eclampsia superimposed on pre-existing hypertension, delivered, with mention of postpartum complication"
        "64273",  # "Pre-eclampsia or eclampsia superimposed on pre-existing hypertension, antepartum condition or complication"
        "64274",  # "Pre-eclampsia or eclampsia superimposed on pre-existing hypertension, postpartum condition or complication"
        "64290",  # "Unspecified hypertension complicating pregnancy, childbirth, or the puerperium, unspecified as to episode of care or not applicable"
        "64291",  # "Unspecified hypertension complicating pregnancy, childbirth, or the puerperium, delivered, with or without mention of antepartum condition"
        "64292",  # "Unspecified hypertension complicating pregnancy, childbirth, or the puerperium, delivered, with mention of postpartum complication"
        "64293",  # "Unspecified hypertension complicating pregnancy, childbirth, or the puerperium, antepartum condition or complication"
        "64294",  # "Unspecified hypertension complicating pregnancy, childbirth, or the puerperium, postpartum condition or complication"
        "99791",  # "Complications affecting other specified body systems, not elsewhere classified, hypertension"
        "G932",  # Benign intracranial hypertension
        "H4005",  # Ocular hypertension
        "H40051",  # "Ocular hypertension, right eye"
        "H40052",  # "Ocular hypertension, left eye"
        "H40053",  # "Ocular hypertension, bilateral"
        "H40059",  # "Ocular hypertension, unspecified eye"
        "I10",  # Essential (primary) hypertension
        "I15",  # Secondary hypertension
        "I150",  # Renovascular hypertension
        "I151",  # Hypertension secondary to other renal disorders
        "I152",  # Hypertension secondary to endocrine disorders
        "I158",  # Other secondary hypertension
        "I159",  # "Secondary hypertension, unspecified"
        "I873",  # Chronic venous hypertension (idiopathic)
        "I8730",  # Chronic venous hypertension (idiopathic) without complications
        "I87301",  # Chronic venous hypertension (idiopathic) without complications of right lower extremity
        "I87302",  # Chronic venous hypertension (idiopathic) without complications of left lower extremity
        "I87303",  # Chronic venous hypertension (idiopathic) without complications of bilateral lower extremity
        "I87309",  # Chronic venous hypertension (idiopathic) without complications of unspecified lower extremity
        "I8731",  # Chronic venous hypertension (idiopathic) with ulcer
        "I87311",  # Chronic venous hypertension (idiopathic) with ulcer of right lower extremity
        "I87312",  # Chronic venous hypertension (idiopathic) with ulcer of left lower extremity
        "I87313",  # Chronic venous hypertension (idiopathic) with ulcer of bilateral lower extremity
        "I87319",  # Chronic venous hypertension (idiopathic) with ulcer of unspecified lower extremity
        "I8732",  # Chronic venous hypertension (idiopathic) with inflammation
        "I87321",  # Chronic venous hypertension (idiopathic) with inflammation of right lower extremity
        "I87322",  # Chronic venous hypertension (idiopathic) with inflammation of left lower extremity
        "I87323",  # Chronic venous hypertension (idiopathic) with inflammation of bilateral lower extremity
        "I87329",  # Chronic venous hypertension (idiopathic) with inflammation of unspecified lower extremity
        "I8733",  # Chronic venous hypertension (idiopathic) with ulcer and inflammation
        "I87331",  # Chronic venous hypertension (idiopathic) with ulcer and inflammation of right lower extremity
        "I87332",  # Chronic venous hypertension (idiopathic) with ulcer and inflammation of left lower extremity
        "I87333",  # Chronic venous hypertension (idiopathic) with ulcer and inflammation of bilateral lower extremity
        "I87339",  # Chronic venous hypertension (idiopathic) with ulcer and inflammation of unspecified lower extremity
        "I8739",  # Chronic venous hypertension (idiopathic) with other complications
        "I87391",  # Chronic venous hypertension (idiopathic) with other complications of right lower extremity
        "I87392",  # Chronic venous hypertension (idiopathic) with other complications of left lower extremity
        "I87393",  # Chronic venous hypertension (idiopathic) with other complications of bilateral lower extremity
        "I87399",  # Chronic venous hypertension (idiopathic) with other complications of unspecified lower extremity
        "O10",  # "Pre-existing hypertension complicating pregnancy, childbirth and the puerperium"
        "O100",  # "Pre-existing essential hypertension complicating pregnancy, childbirth and the puerperium"
        "O1001",  # "Pre-existing essential hypertension complicating pregnancy,"
        "O10011",  # "Pre-existing essential hypertension complicating pregnancy, first trimester"
        "O10012",  # "Pre-existing essential hypertension complicating pregnancy, second trimester"
        "O10013",  # "Pre-existing essential hypertension complicating pregnancy, third trimester"
        "O10019",  # "Pre-existing essential hypertension complicating pregnancy, unspecified trimester"
        "O1002",  # Pre-existing essential hypertension complicating childbirth
        "O1003",  # Pre-existing essential hypertension complicating the puerperium
        "O104",  # "Pre-existing secondary hypertension complicating pregnancy, childbirth and the puerperium"
        "O1041",  # Pre-existing secondary hypertension complicating pregnancy
        "O10411",  # "Pre-existing secondary hypertension complicating pregnancy, first trimester"
        "O10412",  # "Pre-existing secondary hypertension complicating pregnancy, second trimester"
        "O10413",  # "Pre-existing secondary hypertension complicating pregnancy, third trimester"
        "O10419",  # "Pre-existing secondary hypertension complicating pregnancy, unspecified trimester"
        "O1042",  # Pre-existing secondary hypertension complicating childbirth
        "O1043",  # Pre-existing secondary hypertension complicating the puerperium
        "O109",  # "Unspecified pre-existing hypertension complicating pregnancy, childbirth and the puerperium"
        "O1091",  # Unspecified pre-existing hypertension complicating pregnancy
        "O10911",  # "Unspecified pre-existing hypertension complicating pregnancy, first trimester"
        "O10912",  # "Unspecified pre-existing hypertension complicating pregnancy, second trimester"
        "O10913",  # "Unspecified pre-existing hypertension complicating pregnancy, third trimester"
        "O10919",  # "Unspecified pre-existing hypertension complicating pregnancy, unspecified trimester"
        "O1092",  # Unspecified pre-existing hypertension complicating childbirth
        "O1093",  # Unspecified pre-existing hypertension complicating the puerperium
        "O11",  # Pre-existing hypertension with pre-eclampsia
        "O111",  # "Pre-existing hypertension with pre-eclampsia, first trimester"
        "O112",  # "Pre-existing hypertension with pre-eclampsia, second trimester"
        "O113",  # "Pre-existing hypertension with pre-eclampsia, third trimester"
        "O114",  # "Pre-existing hypertension with pre-eclampsia, complicating childbirth"
        "O115",  # "Pre-existing hypertension with pre-eclampsia, complicating the puerperium"
        "O119",  # "Pre-existing hypertension with pre-eclampsia, unspecified trimester"
        "O13",  # Gestational [pregnancy-induced] hypertension without significant proteinuria
        "O131",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, first trimester"
        "O132",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, second trimester"
        "O133",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, third trimester"
        "O134",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, complicating childbirth"
        "O135",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, complicating the puerperium"
        "O139",  # "Gestational [pregnancy-induced] hypertension without significant proteinuria, unspecified trimester"
        "O16",  # Unspecified maternal hypertension
        "O161",  # "Unspecified maternal hypertension, first trimester"
        "O162",  # "Unspecified maternal hypertension, second trimester"
        "O163",  # "Unspecified maternal hypertension, third trimester"
        "O164",  # "Unspecified maternal hypertension, complicating childbirth"
        "O165",  # "Unspecified maternal hypertension, complicating the puerperium"
        "O169",  # "Unspecified maternal hypertension, unspecified trimester"
        "R030",  # "Elevated blood-pressure reading, without diagnosis of hypertension"
    ]

    df = getTargetPatientIcd()
    df = df[df["icd_code"].isin(codes)]

    df = df.copy()
    df["hypertension"] = True

    return (
        df[["hadm_id", "hypertension"]]
        .groupby("hadm_id")["hypertension"]
        .any()
        .reset_index()
    )


def getUTI():
    """Urinary tract infection.

    True/False

    Returns:
        pandas.DataFrame: ["hadm_id", "uti"]
    """

    codes = [
        "5990",  # "Urinary tract infection, site not specified"
        "77182",  # Urinary tract infection of newborn
        "N390",  # "Urinary tract infection, site not specified"
        "O0338",  # Urinary tract infection following incomplete spontaneous abortion
        "O0388",  # Urinary tract infection following complete or unspecified spontaneous abortion
        "O0488",  # Urinary tract infection following (induced) termination of pregnancy
        "O0738",  # Urinary tract infection following failed attempted termination of pregnancy
        "O0883",  # Urinary tract infection following an ectopic and molar pregnancy
        "O862",  # Urinary tract infection following delivery
        "O8620",  # "Urinary tract infection following delivery, unspecified"
        "O8629",  # Other urinary tract infection following delivery
        "P393",  # Neonatal urinary tract infection
    ]

    df = getTargetPatientIcd()
    df = df[df["icd_code"].isin(codes)]

    df = df.copy()
    df["uti"] = True

    return df[["hadm_id", "uti"]].groupby("hadm_id")["uti"].any().reset_index()


def getChronicPulmonaryDisease():
    """True/False

    Returns:
        pandas.DataFrame: ["hadm_id", "chronic_pulmonary_disease"]
    """

    df = charlson.runSql()
    df["chronic_pulmonary_disease"] = df["chronic_pulmonary_disease"].astype(bool)

    return (
        df[["hadm_id", "chronic_pulmonary_disease"]]
        .groupby("hadm_id")["chronic_pulmonary_disease"]
        .any()
        .reset_index()
    )
