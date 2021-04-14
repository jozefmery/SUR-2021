
from keras.models import load_model
import os

def main():

  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

  # load the model

  # do not compile, as pre-trained model has no training config
  model = load_model("../model/face/pretrained/facenet_keras.h5", compile=False)
 
  # compile manually
  # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
  model.compile()

  ####################
  # TESTING CODE
  # summarize input and output shape
  print(model.inputs)
  print(model.outputs)
  ####################

if __name__ == "__main__":
  main()