import io
from flask import Flask, jsonify, request
import joblib
from matplotlib import pyplot as plt
from pandas import Timestamp
from tinydb import TinyDB
from tinydb.queries import Query
from constants import CATEGORICAL_MEASURES, TEMP_PATH
from utils.class_patient import Patient
from utils.class_voter import Voter
from lime.lime_tabular import LimeTabularExplainer

from utils.prepare_data import encodeCategoricalData, getMonitoredPatients


app = Flask(__name__)

models = [joblib.load(TEMP_PATH / f"tabpfn_last_{i}.pkl") for i in range(5)]
voter = Voter(models)

patients = getMonitoredPatients()
dfPatients = patients.getMeasuresBetween(how="last", getUntilAkiPositive=True)
dfPatients, *_ = encodeCategoricalData(dfPatients, dfPatients)
categoricalIdx = [dfPatients.columns.get_loc(c) for c in dfPatients.columns if c.startswith(tuple(CATEGORICAL_MEASURES))]
lime = LimeTabularExplainer(dfPatients.to_numpy(), mode="classification", categorical_features=categoricalIdx, feature_names=dfPatients.columns)

db = TinyDB(TEMP_PATH / "db.json")


def predict(item_id):
    pQuery = Query()
    pStr = db.search(pQuery.stay_id == item_id)[0]

    patient = Patient.fromJson(pStr)
    dfPatient = patient.getMeasuresBetween(how="last")

    pred = voter.predict_proba(dfPatient)

    # lime explain
    exp = lime.explain_instance(dfPatient, voter.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure()

    return pred, fig 


def upsertPatient(pId, mName, mValue, mTime):
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


@app.route("/<int:item_id>", methods=["GET"])
def get_items(item_id):
    pred, fig = predict(item_id)
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
    
    if mName and mValue:
        mValue = float(mValue)
        if not mTime:
            patient = upsertPatient(pId, mName, mValue, None)
        else:
            patient = upsertPatient(pId, mName, mValue, mTime)
    return jsonify(patient.getMeasuresBetween(how="last")), 201


def runRestServer():
    app.run(port=5000)
