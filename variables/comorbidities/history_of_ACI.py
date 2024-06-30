import pandas as pd

from constants import MIMIC_PATH
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu


def get():
    """_summary_

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    fullCodes = [
        "34660",
        "34661",
        "34662",
        "34663",
        "43301",
        "43311",
        "43321",
        "43331",
        "43381",
        "43391",
        "43401",
        "43411",
        "43491",
        "G436",
        "G4360",
        "G43601",
        "G43609",
        "G4361",
        "G43611",
        "G43619",
        "I63",
        "I630",
        "I6300",
        "I6301",
        "I63011",
        "I63012",
        "I63013",
        "I63019",
        "I6302",
        "I6303",
        "I63031",
        "I63032",
        "I63033",
        "I63039",
        "I6309",
        "I631",
        "I6310",
        "I6311",
        "I63111",
        "I63112",
        "I63113",
        "I63119",
        "I6312",
        "I6313",
        "I63131",
        "I63132",
        "I63133",
        "I63139",
        "I6319",
        "I632",
        "I6320",
        "I6321",
        "I63211",
        "I63212",
        "I63213",
        "I63219",
        "I6322",
        "I6323",
        "I63231",
        "I63232",
        "I63233",
        "I63239",
        "I6329",
        "I633",
        "I6330",
        "I6331",
        "I63311",
        "I63312",
        "I63313",
        "I63319",
        "I6332",
        "I63321",
        "I63322",
        "I63323",
        "I63329",
        "I6333",
        "I63331",
        "I63332",
        "I63333",
        "I63339",
        "I6334",
        "I63341",
        "I63342",
        "I63343",
        "I63349",
        "I6339",
        "I634",
        "I6340",
        "I6341",
        "I63411",
        "I63412",
        "I63413",  # Cerebral infarction due to embolism of bilateral middle cerebral arteries
        "I63419",  # Cerebral infarction due to embolism of unspecified middle cerebral artery
        "I6342",  # Cerebral infarction due to embolism of anterior cerebral artery
        "I63421",  # Cerebral infarction due to embolism of right anterior cerebral artery
        "I63422",  # Cerebral infarction due to embolism of left anterior cerebral artery
        "I63423",  # Cerebral infarction due to embolism of bilateral anterior cerebral arteries
        "I63429",  # Cerebral infarction due to embolism of unspecified anterior cerebral artery
        "I6343",  # Cerebral infarction due to embolism of posterior cerebral artery
        "I63431",  # Cerebral infarction due to embolism of right posterior cerebral artery
        "I63432",  # Cerebral infarction due to embolism of left posterior cerebral artery
        "I63433",  # Cerebral infarction due to embolism of bilateral posterior cerebral arteries
        "I63439",  # Cerebral infarction due to embolism of unspecified posterior cerebral artery
        "I6344",  # Cerebral infarction due to embolism of cerebellar artery
        "I63441",  # Cerebral infarction due to embolism of right cerebellar artery
        "I63442",  # Cerebral infarction due to embolism of left cerebellar artery
        "I63443",  # Cerebral infarction due to embolism of bilateral cerebellar arteries
        "I63449",  # Cerebral infarction due to embolism of unspecified cerebellar artery
        "I6349",  # Cerebral infarction due to embolism of other cerebral artery
        "I635",  # Cerebral infarction due to unspecified occlusion or stenosis of cerebral arteries
        "I6350",  # Cerebral infarction due to unspecified occlusion or stenosis of unspecified cerebral artery
        "I6351",  # Cerebral infarction due to unspecified occlusion or stenosis of middle cerebral artery
        "I63511",  # Cerebral infarction due to unspecified occlusion or stenosis of right middle cerebral artery
        "I63512",  # Cerebral infarction due to unspecified occlusion or stenosis of left middle cerebral artery
        "I63513",  # Cerebral infarction due to unspecified occlusion or stenosis of bilateral middle cerebral arteries
        "I63519",  # Cerebral infarction due to unspecified occlusion or stenosis of unspecified middle cerebral artery
        "I6352",  # Cerebral infarction due to unspecified occlusion or stenosis of anterior cerebral artery
        "I63521",  # Cerebral infarction due to unspecified occlusion or stenosis of right anterior cerebral artery
        "I63522",  # Cerebral infarction due to unspecified occlusion or stenosis of left anterior cerebral artery
        "I63523",  # Cerebral infarction due to unspecified occlusion or stenosis of bilateral anterior cerebral arteries
        "I63529",  # Cerebral infarction due to unspecified occlusion or stenosis of unspecified anterior cerebral artery
        "I6353",  # Cerebral infarction due to unspecified occlusion or stenosis of posterior cerebral artery
        "I63531",  # Cerebral infarction due to unspecified occlusion or stenosis of right posterior cerebral artery
        "I63532",  # Cerebral infarction due to unspecified occlusion or stenosis of left posterior cerebral artery
        "I63533",  # Cerebral infarction due to unspecified occlusion or stenosis of bilateral posterior cerebral arteries
        "I63539",  # Cerebral infarction due to unspecified occlusion or stenosis of unspecified posterior cerebral artery
        "I6354",  # Cerebral infarction due to unspecified occlusion or stenosis of cerebellar artery
        "I63541",  # Cerebral infarction due to unspecified occlusion or stenosis of right cerebellar artery
        "I63542",  # Cerebral infarction due to unspecified occlusion or stenosis of left cerebellar artery
        "I63543",  # Cerebral infarction due to unspecified occlusion or stenosis of bilateral cerebellar arteries
        "I63549",  # Cerebral infarction due to unspecified occlusion or stenosis of unspecified cerebellar artery
        "I6359",  # Cerebral infarction due to unspecified occlusion or stenosis of other cerebral artery
        "I636",  # "Cerebral infarction due to cerebral venous thrombosis, nonpyogenic"
        "I638",  # Other cerebral infarction
        "I6381",  # Other cerebral infarction due to occlusion or stenosis of small artery
        "I6389",  # Other cerebral infarction
        "I639",  # "Cerebral infarction, unspecified"
        "I693",  # Sequelae of cerebral infarction
        "I6930",  # Unspecified sequelae of cerebral infarction
        "I6931",  # Cognitive deficits following cerebral infarction
        "I69310",  # Attention and concentration deficit following cerebral infarction
        "I69311",  # Memory deficit following cerebral infarction
        "I69312",  # Visuospatial deficit and spatial neglect following cerebral infarction
        "I69313",  # Psychomotor deficit following cerebral infarction
        "I69314",  # Frontal lobe and executive function deficit following cerebral infarction
        "I69315",  # Cognitive social or emotional deficit following cerebral infarction
        "I69318",  # Other symptoms and signs involving cognitive functions following cerebral infarction
        "I69319",  # Unspecified symptoms and signs involving cognitive functions following cerebral infarction
        "I6932",  # Speech and language deficits following cerebral infarction
        "I69320",  # Aphasia following cerebral infarction
        "I69321",  # Dysphasia following cerebral infarction
        "I69322",  # Dysarthria following cerebral infarction
        "I69323",  # Fluency disorder following cerebral infarction
        "I69328",  # Other speech and language deficits following cerebral infarction
        "I6933",  # Monoplegia of upper limb following cerebral infarction
        "I69331",  # Monoplegia of upper limb following cerebral infarction affecting right dominant side
        "I69332",  # Monoplegia of upper limb following cerebral infarction affecting left dominant side
        "I69333",  # Monoplegia of upper limb following cerebral infarction affecting right non-dominant side
        "I69334",  # Monoplegia of upper limb following cerebral infarction affecting left non-dominant side
        "I69339",  # Monoplegia of upper limb following cerebral infarction affecting unspecified side
        "I6934",  # Monoplegia of lower limb following cerebral infarction
        "I69341",  # Monoplegia of lower limb following cerebral infarction affecting right dominant side
        "I69342",  # Monoplegia of lower limb following cerebral infarction affecting left dominant side
        "I69343",  # Monoplegia of lower limb following cerebral infarction affecting right non-dominant side
        "I69344",  # Monoplegia of lower limb following cerebral infarction affecting left non-dominant side
        "I69349",  # Monoplegia of lower limb following cerebral infarction affecting unspecified side
        "I6935",  # Hemiplegia and hemiparesis following cerebral infarction
        "I69351",  # Hemiplegia and hemiparesis following cerebral infarction affecting right dominant side
        "I69352",  # Hemiplegia and hemiparesis following cerebral infarction affecting left dominant side
        "I69353",  # Hemiplegia and hemiparesis following cerebral infarction affecting right non-dominant side
        "I69354",  # Hemiplegia and hemiparesis following cerebral infarction affecting left non-dominant side
        "I69359",  # Hemiplegia and hemiparesis following cerebral infarction affecting unspecified side
        "I6936",  # Other paralytic syndrome following cerebral infarction
        "I69361",  # Other paralytic syndrome following cerebral infarction affecting right dominant side
        "I69362",  # Other paralytic syndrome following cerebral infarction affecting left dominant side
        "I69363",  # Other paralytic syndrome following cerebral infarction affecting right non-dominant side
        "I69364",  # Other paralytic syndrome following cerebral infarction affecting left non-dominant side
        "I69365",  # "Other paralytic syndrome following cerebral infarction, bilateral"
        "I69369",  # Other paralytic syndrome following cerebral infarction affecting unspecified side
        "I6939",  # Other sequelae of cerebral infarction
        "I69390",  # Apraxia following cerebral infarction
        "I69391",  # Dysphagia following cerebral infarction
        "I69392",  # Facial weakness following cerebral infarction
        "I69393",  # Ataxia following cerebral infarction
        "I69398",  # Other sequelae of cerebral infarction
    ]

    dfPatIcu = getTargetPatientIcu()
    dfPatIcu = dfPatIcu[["subject_id", "hadm_id"]]
    dfPatIcu.sort_values("subject_id", inplace=True)

    patients = set(dfPatIcu["subject_id"])

    dfIcd = pd.read_csv(MIMIC_PATH / "hosp" / "diagnoses_icd.csv")
    dfIcd = dfIcd[dfIcd["subject_id"].isin(patients)]

    dfIcdPrevHadm = dfIcd.merge(dfPatIcu, on="subject_id", suffixes=("_all", "_target"))
    dfIcdPrevHadm = dfIcdPrevHadm[
        dfIcdPrevHadm["hadm_id_all"] <= dfIcdPrevHadm["hadm_id_target"]
    ]
    dfIcdPrevHadm = dfIcdPrevHadm.rename(columns={"hadm_id_all": "hadm_id"})

    dfIcuPrevAmi = dfIcdPrevHadm[dfIcdPrevHadm["icd_code"].isin(fullCodes)]

    df = pd.DataFrame(
        {
            "hadm_id": list(set(dfIcuPrevAmi["hadm_id_target"])),
            "history_aci": True,
        }
    )

    return df.sort_values("hadm_id")
