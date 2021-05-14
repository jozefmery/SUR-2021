# File:     face_classifier.py
# Date:     13.05.2021
# Project:  SUR-2021
# Authors:  Dovičic Denis     - xdovic01@vutbr.cz
#           Hudecová Patrícia - xhudec30@vutbr.cz
#           Méry Jozef        - xmeryj00@vutbr.cz
# Description: Face classification related items.  

import keras.models
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# implementation adopted from the following site:
# https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

def _load_facenet(path: str):
  # do not compile as no compilation configuration is provided
  model = keras.models.load_model(path, compile=False)
  # compile manually
  model.compile()
  return model

def _extract_embeddings(model, image_data):
  # ensure correct data type
  image_data = image_data.astype("float32")
  # prepare image data for facenet prediction (standardize)
  mean, std = image_data.mean(), image_data.std()
  image_data = (image_data - mean) / std
  samples = np.expand_dims(image_data, axis=0)
  # predict embeddings using the pre-trained facenet model
  return model.predict(samples)[0]

def _normalize_embeddings(embeddings):
  # transform each embedding vector to a unit vector
  return Normalizer(norm="l2").transform(embeddings)

def _images_to_embeddings(facenet_path: str, images):
  model = _load_facenet(facenet_path)
  embeddings = np.array([_extract_embeddings(model, image) for image in images])
  return _normalize_embeddings(embeddings)

def _facenet_model_path(models_base_path: str):
  return os.path.join(models_base_path, "facenet", "facenet.h5")

def train_svm(train_data, dev_data, models_base_path):
  facenet_path = _facenet_model_path(models_base_path)
  # train the model using the training data
  embeddings = _images_to_embeddings(facenet_path, train_data[0])
  model = SVC(kernel="linear")
  model.fit(embeddings, train_data[1])
  # evaluate the trained model using the dev data
  prediction = predict(model, dev_data[0], models_base_path)
  score = accuracy_score(dev_data[1], prediction)
  
  return model, score

def predict(model, images, models_base_path):
  facenet_path = _facenet_model_path(models_base_path)
  embeddings = _images_to_embeddings(facenet_path, images)
  return model.predict(embeddings)