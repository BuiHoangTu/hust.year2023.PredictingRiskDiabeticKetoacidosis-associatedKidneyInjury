import io
from flask import Flask, jsonify, request
import joblib
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import Timestamp
from tinydb import TinyDB
from tinydb.queries import Query
from constants import CATEGORICAL_MEASURES, TEMP_PATH
from utils.class_patient import Patient
from utils.class_voter import Voter
from lime.lime_tabular import LimeTabularExplainer

from utils.prepare_data import NormalizeData, encodeCategoricalData, getMonitoredPatients


app = Flask(__name__)

models = [joblib.load(TEMP_PATH / f"tabpfn_last_{i}.pkl") for i in range(5)]
voter = Voter(models) # type: ignore

patients = getMonitoredPatients()
dfPatients = patients.getMeasuresBetween(how="last", getUntilAkiPositive=True)
dfPatients, *_ = encodeCategoricalData(dfPatients, dfPatients)
categoricalIdx = [dfPatients.columns.get_loc(c) for c in dfPatients.columns if c.startswith(tuple(CATEGORICAL_MEASURES))]
lime = LimeTabularExplainer(dfPatients.to_numpy(), mode="classification", categorical_features=categoricalIdx, feature_names=dfPatients.columns)


db = TinyDB(TEMP_PATH / "db.json")


def predict(item_id):
    item_id = int(item_id)

    pQuery = Query()
    pStr = db.search(pQuery.stay_id == item_id)[0]

    patient = Patient.fromJson(pStr)
    dfPredict = patient.getMeasuresBetween(how="last")
    dfPredict = dfPredict.fillna(0)

    pred = voter.predict_proba(dfPredict)

    fig = None
    # # lime explain
    # # fill dfPred with columns missing from dfAll
    # dfPredict = dfPredict.reindex(columns=dfPatients.columns, fill_value=0)
    # dfPredict = dfPredict.fillna(0)
    # print(dfPredict)
    # exp = lime.explain_instance((dfPredict.iloc[0]), voter.predict_proba, num_features=10)
    # fig = exp.as_pyplot_figure()
    return pred, fig    


def upsertPatient(pId, mName, mValue, mTime):
    pId = int(pId)
    
    pQuery = Query()
    pStr = db.search(pQuery.stay_id == pId)

    if len(pStr) == 0:
        patient = Patient(
            0,
            0,
            pId,
            Timestamp.now(),
            False,
        )
        db.insert(patient.toJson())
    else:
        patient = Patient.fromJson(pStr[0])

    patient.putMeasure(mName, mTime, mValue, existingTypeIncompatible="skip")

    db.update(patient.toJson(), pQuery.stay_id == pId)
    
    return patient

def getPatient(pId):
    pId = int(pId)
    
    pQuery = Query()
    pStr = db.search(pQuery.stay_id == pId)
    
    if len(pStr) == 0:
        return None
    else:
        return Patient.fromJson(pStr[0])


@app.route("/<int:item_id>", methods=["GET"])
def get_items(item_id):
    item_id = int(item_id)
    
    pred, fig = predict(item_id)
    if fig is None:
        print(pred)
        return jsonify({"prediction": float(pred[0][0])})
    imgIo = io.BytesIO()
    fig.savefig(imgIo, format="png")
    imgIo.seek(0)
    plt.close(fig)

    return jsonify({
        "prediction": pred, 
        "explanation": imgIo.getvalue().decode("latin1")
    })


@app.route("/", methods=["POST"])
def create_item():
    new_item = request.get_json()
    
    pId = new_item["stay_id"]
    mName: str = new_item["measurement"]
    mValue = new_item["value"]
    mTime: str = new_item["time"]
    
    pId = int(pId)
    
    if mName and mValue:
        mValue = float(mValue)
        if not mTime:
            patient = upsertPatient(pId, mName, mValue, None)
        else:
            patient = upsertPatient(pId, mName, mValue, mTime)
            
    else:
        patient = getPatient(pId)
        if patient is None:
            return jsonify({"error": "Patient not found"}), 404
    
    dfPE = patient.getMeasuresBetween(how="last")
    dictPE = dfPE.iloc[0].to_dict()
    dictPE = {str(k): str(v) for k, v in dictPE.items()}
    
    return jsonify(dictPE), 201


def runRestServer():
    app.run(port=5000, debug=True)
