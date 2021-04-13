
from keras.models import load_model

def main():

  # load the model
  model = load_model("facenet_keras.h5")
  # summarize input and output shape
  print(model.inputs)
  print(model.outputs)

if __name__ == "__main__":
  main()