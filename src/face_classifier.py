import keras.models
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def _load_model(path: str):
  model = keras.models.load_model(path, compile=False)
  model.compile()
  return model

def _extract_embeddings(model, image_data):
  image_data = image_data.astype("float32")
  mean, std = image_data.mean(), image_data.std()
  image_data = (image_data - mean) / std
  samples = np.expand_dims(image_data, axis=0)
  return model.predict(samples)[0]

def _normalize_embeddings(embeddings):
  return Normalizer(norm="l2").transform(embeddings)

def _images_to_embeddings(facenet_path: str, images):
  model = _load_model(facenet_path)
  embeddings = np.array([_extract_embeddings(model, image) for image in images])
  return _normalize_embeddings(embeddings)

def train_svm(images, targets, path):
  model = SVC(kernel="linear")
  model.fit(images, targets)
  return model

def evaluate(model):
  pass