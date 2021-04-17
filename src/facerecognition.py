from common import shut_up_tensorflow_thanks

# make tensorflow shut up
shut_up_tensorflow_thanks()

from os.path import join as joinpath
from os.path import dirname
import os
# import os
from keras.models import load_model

def face_model_dir():
  return joinpath(dirname(__file__), os.pardir, "model", "face")

def pretrained_face_model_path():
  return joinpath(face_model_dir(), "pretrained", "facenet_keras.h5")

def enable_xla_devices():
  # no idea what this is, but tensorflow is happier
  # happy tensorflow === happy life
  # https://www.tensorflow.org/xla
  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

def main():

  ####################
  # TESTING CODE
  # do not directly run this file

  enable_xla_devices()

  # load the model
  # do not compile, as pre-trained model has no training config
  model = load_model(pretrained_face_model_path(), compile=False)
 
  # compile manually
  # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
  model.compile()

  # summarize input and output shape
  print(model.inputs)
  print(model.outputs)
  ####################

if __name__ == "__main__":
  main()