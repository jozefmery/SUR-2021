from utils import set_tf_loglevel_warn

# needs to be called before importing tensorflow
set_tf_loglevel_warn()

import utils
import face_classifier
import dataloader
import argparse
import os
import sys

# constants
DEFAULT_DATASET_PATH  = os.path.join(os.path.dirname(sys.argv[0]), os.path.pardir, "dataset")
DEFAULT_MODELS_PATH   = os.path.join(os.path.dirname(sys.argv[0]), os.path.pardir, "model")
DEFAULT_RESULT_PATH   = os.path.join(os.path.dirname(sys.argv[0]), os.path.pardir, "results")

def parse_arguments():
  parser = argparse.ArgumentParser(
    description="SUR-2021 Face and Voice recognizer",
    epilog= "Authors:\n"
            "Denis Dovičic - xdovic01@vutbr.cz\n"
            "Patrícia Hudecová - xhudec30@vutbr.cz\n"
            "Jozef Méry - xmeryj00@vutbr.cz",
    # use RawTextHelpFormatter to correctly interpret newlines
    formatter_class=argparse.RawTextHelpFormatter,
    # disable slightly confusing abbreviations
    allow_abbrev=False,
    add_help=False
  )

  parser.add_argument("system", choices=["face", "voice"], help="Choose a system to execute an action on.")
  
  # override default help for consistency
  parser.add_argument("-h", "--help", action="help",
    default=argparse.SUPPRESS,
    help="Display this help message and exit.")
  
  parser.add_argument("--models", action="store",
    default=DEFAULT_MODELS_PATH,
    metavar="path",
    help="Path to the models base directory.")
  
  parser.add_argument("--dataset", action="store",
    default=DEFAULT_DATASET_PATH,
    metavar="path",
    help="Path to the dataset base directory.")

  parser.add_argument("--results", action="store",
    default=DEFAULT_RESULT_PATH,
    metavar="path",
    help="Directory, where the results will be saved if eval action is chosen.")

  parser.add_argument("--action", default="eval", choices=["eval", "train"], help="Evaluate an existing model (default) or begin training a new model.")
  
  args = parser.parse_args()
  # convert string to enum variant
  args.system   = dataloader.System(args.system)
  return args

def train_face(train_data, dev_data, models_path):
  # train model
  model, score = face_classifier.train_svm(train_data, dev_data, models_path)
  # save trained model
  utils.save_model(os.path.join(models_path, "face", "model.svm"), model)
  return score

def train_voice(train_data, dev_data, models_path):
  # TODO 
  return 0

def eval_face(data, models_path):
  model = utils.load_model(os.path.join(models_path, "face", "model.svm"))
  return face_classifier.predict(model, data, models_path)

def eval_voice(data, models_path):
  # TODO
  pass

SYS_TRAINING_MAPPER = {

  dataloader.System.FACE:   train_face,
  dataloader.System.VOICE:  train_voice,
}

SYS_EVAL_MAPPER = {

  dataloader.System.FACE:   eval_face,
  dataloader.System.VOICE:  eval_voice,
}

def train(args):
  print("Training the " + args.system.value + " system model...")
  # load training and dev data
  train_data = dataloader.load(args.dataset, dataloader.Category.TRAIN, args.system)
  dev_data = dataloader.load(args.dataset, dataloader.Category.DEV, args.system)
  # train the model and get its score on the dev data category
  score = SYS_TRAINING_MAPPER[args.system](train_data, dev_data, args.models)
  print("Finished training the " + args.system.value + " system, score on dev data: {:.2f}%".format(score * 100))

def eval(args):
  print("Evaluating the " + args.system.value + " system...")
  # load data and make predictions
  data, ids = dataloader.load(args.dataset, dataloader.Category.EVAL, args.system)
  predictions = SYS_EVAL_MAPPER[args.system](data, args.models)
  # write results
  results_path = os.path.join(args.results, args.system.value, "evaluation.txt")
  utils.write_results(results_path, ids, predictions)

  print("Finished evaluating the " + args.system.value + " system.")

ACTION_MAPPER = {

  "eval": eval,
  "train": train
}

def main():
  
  try:

    utils.enable_xla_devices()

    args = parse_arguments()

    ACTION_MAPPER[args.action](args)
    
  except RuntimeError as e:
    
    print("Error: " + str(e))

if __name__ == "__main__":
  main()