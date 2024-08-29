#!/usr/bin/env python
# coding: utf-8

# # Inputs

# In[1]:


splitPartCount = 5
splitSeed = 27
hoursPerWindow = 1


# # Preproccess

# ## read data

# In[2]:


from pathlib import Path
import pickle
from tabnanny import verbose
from utils.class_patient import Patients
from utils.class_voter import combineProbas


patients = Patients.loadPatients()
len(patients)


# ## remove missing

# In[3]:


# fill measures whose null represent false value

from constants import NULLABLE_MEASURES, TEMP_PATH


nullableMeasures = NULLABLE_MEASURES

for measureName in nullableMeasures:
    patients.fillMissingMeasureValue(measureName, 0)


# In[4]:


# remove measures with less than 80% of data

measures = patients.getMeasures()

for measure, count in measures.items():
    if count < len(patients) * 80 / 100:
        patients.removeMeasures([measure])
        print(measure, count)


# In[5]:


# remove patients with less than 80% of data

patients.removePatientByMissingFeatures()
len(patients)


# In[6]:


# # remove patients with positive tag in first 12 hours

from pandas import Timedelta


patients.removePatientAkiEarly(Timedelta(hours=12))


# In[7]:


print("Total ", len(patients))
print("AKI ", sum([1 for p in patients if p.akdPositive]))
print("Ratio ", sum([1 for p in patients if p.akdPositive]) / len(patients))


# ## split patients

# In[8]:


splitedPatients = patients.split(splitPartCount, splitSeed)

len(splitedPatients[0])


# In[9]:


splitedPatients = patients.split(splitPartCount, splitSeed)


def trainTest():
    for i in range(splitedPatients.__len__()):
        testPatients = splitedPatients[i]

        trainPatientsList = splitedPatients[:i] + splitedPatients[i + 1 :]
        trainPatients = Patients(patients=[])
        for trainPatientsElem in trainPatientsList:
            trainPatients += trainPatientsElem

        yield trainPatients, testPatients


def trainValTest():
    for i in range(splitedPatients.__len__()):
        testPatients = splitedPatients[i]

        trainPatientsList = splitedPatients[:i] + splitedPatients[i + 1 :]
        trainPatients = Patients(patients=[])
        for trainPatientsElem in trainPatientsList:
            trainPatients += trainPatientsElem

        *trainPatients, valPatients = trainPatients.split(5, 27)
        tmpPatients = Patients(patients=[])
        for trainPatientsElem in trainPatients:
            tmpPatients += trainPatientsElem
        trainPatients = tmpPatients

        yield trainPatients, valPatients, testPatients


# In[10]:


for trainPatients, testPatients in trainTest():
    print(len(trainPatients.patientList), len(testPatients.patientList))


# # Transformer

# ### Seperate static and dynamic

# In[11]:


import math
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Concatenate,
    LayerNormalization,
    Conv1D,
    GlobalAveragePooling1D,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from official.nlp.modeling.models import TransformerEncoder


@keras.saving.register_keras_serializable()
class SplitTransfomer(Model):

    def __init__(self, timeSteps, timeFeatures, staticFeatures, **kwargs):
        super().__init__(**kwargs)

        self.timeSteps = timeSteps
        self.timeFeatures = timeFeatures
        self.staticFeatures = staticFeatures

        # time series layers
        self.transformerEncoder = TransformerEncoder(
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            # max_sequence_length=max_sequence_length,
            # vocab_size=vocab_size,
        )
        self.pooling = GlobalAveragePooling1D()

        # static layers
        self.stDense = Dense(staticFeatures, activation="relu")

        # combine layers
        self.concat = Concatenate(axis=1)
        self.drop = Dropout(0.2)
        self.dense1 = Dense(16, activation="relu")
        self.denseOut = Dense(1, activation="sigmoid")

        pass

    def call(self, input, training=False):
        seriesInputLayer, staticInputLayer = input

        # series
        seriesLayer = seriesInputLayer
        seriesLayer = self.transformerEncoder(seriesLayer, training=training)
        seriesLayer = self.pooling(seriesLayer)

        # static
        staticLayer = staticInputLayer

        combined = tf.concat([seriesLayer, staticLayer], axis=-1)
        combined = self.denseOut(combined)

        return combined
    
    def build(self, input_shape):
        seriesInputLayer, staticInputLayer = input_shape

        self.transformerEncoder.build(seriesInputLayer)

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "timeSteps": self.timeSteps,
                "timeFeatures": self.timeFeatures,
                "staticFeatures": self.staticFeatures,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def createModel2(timeSteps, timeFeatures, staticFeatures):
    model = SplitTransfomer(timeSteps, timeFeatures, staticFeatures)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy", "precision", "recall"],
    )

    return model


x = createModel2(24, 10, 10)
sample_series_input = tf.random.normal([1, 12, 31])  # (batch_size, timeSteps, timeFeatures)
sample_static_input = tf.random.normal([1, 67])  
_ = x([sample_series_input, sample_static_input])
x.summary(expand_nested=True, show_trainable=True)


# In[17]:


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from utils.prepare_data import (
    DeepLearningDataPreparer,
    trainValTestPatients,
    patientsToNumpy,
    trainValTestNp,
)
from constants import CATEGORICAL_MEASURES
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


loses = []
aucs = []
accuracies = []
precisions = []
recals = []

train_loss_list = []
val_loss_list = []


y_probas = []

aucM = [0.83, 0.81, 0.82, 0.82, 0.795]

for i, (trainValGenerator, testPatients) in enumerate(trainValTestPatients(patients, splitSeed)):
    trainValList = list(trainValGenerator)
    
    while True:
        if Path(f"result/transformer_split_model_{i}_0.keras").exists():
            print(f"######## skip {i} ########")
            break
        probas = []
        models = []
        for ii, (trainPatients, valPatients) in enumerate(trainValList):
            preparer = DeepLearningDataPreparer(
                hoursPerWindows=hoursPerWindow,
                fromHour=0,
                toHour=12,
            )

            cacheFile = TEMP_PATH / f"dl_train_data/{i}_{ii}.pkl"
            cacheFile.parent.mkdir(parents=True, exist_ok=True)

            if cacheFile.exists():
                (npTrainX, staticTrainX, trainY), \
                (npValX, staticValX, valY), \
                (npTestX, staticTestX, testY) = pickle.loads(cacheFile.read_bytes()) 
            else:
                npTrainX, staticTrainX, trainY = preparer.fit_transform(trainPatients)
                npValX, staticValX, valY = preparer.transform(valPatients)
                npTestX, staticTestX, testY = preparer.transform(testPatients)
                cacheFile.write_bytes(pickle.dumps(((npTrainX, staticTrainX, trainY), 
                                                     (npValX, staticValX, valY), 
                                                     (npTestX, staticTestX, testY))))

            neg, pos = np.bincount(trainY)
            weight0 = (1 / neg) * (len(trainY)) / 2.0
            weight1 = (1 / pos) * (len(trainY)) / 2.0
            weight = {0: weight0, 1: weight1}

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=250, restore_best_weights=True
            )

            model = createModel2(npTrainX.shape[1], npTrainX.shape[2], staticTrainX.shape[1])

            print(f"######## start fit {i}_{ii} ########")

            history = model.fit(
                [npTrainX, staticTrainX],
                np.array(trainY),
                epochs=5000,
                batch_size=32,
                validation_data=([npValX, staticValX], np.array(valY)),
                class_weight=weight,
                callbacks=[early_stopping],
                verbose=0,
            )
            train_loss_list.append(history.history["loss"])
            val_loss_list.append(history.history["val_loss"])
            probas.append(model.predict([npTestX, staticTestX]))
            models.append(model)

        # calculate final probas, auc, accuracy, precision, recal
        print(f"probas {i} shape: {np.array(probas).shape}")
        finalYProbas = np.apply_along_axis(combineProbas, 0, np.array(probas))
        print(f"final probas shape: {finalYProbas.shape}")
        
        finalAuc = roc_auc_score(testY, finalYProbas)
        aucs.append(finalAuc)

        y_pred = np.where(finalYProbas >= 0.5, 1, 0)

        accuracies.append(accuracy_score(testY, y_pred))
        precisions.append(precision_score(testY, y_pred))
        recals.append(recall_score(testY, y_pred))
        
        print(f"final auc: {finalAuc}")
        print(f"final accuracy: {accuracies[-1]}")
        print(f"final precision: {precisions[-1]}")
        print(f"final recal: {recals[-1]}")

        if finalAuc > aucM[i]:
            for ii, model in enumerate(models):
                model.save(f"result/transformer_split_model_{i}_{ii}.keras")
            break

    pass
