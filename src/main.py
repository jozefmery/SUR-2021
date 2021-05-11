
from utils import set_tf_loglevel_warn, enable_xla_devices

# needs to be called before importing tensorflow
set_tf_loglevel_warn()

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
  parser.add_argument("--category", default="eval", choices=["eval", "train", "dev"], help="Choose dataset category to perform the selected action on. Eval by default. \nNote that training is not possible on the eval category.")
  
  return parser.parse_args()
 
def main():
  
  try:
    enable_xla_devices()

    args = parse_arguments()
    print(args)
    # data, targets = dataloader.load(DEFAULT_DATASET_PATH, dataloader.DataClass.TRAIN, dataloader.System.FACE)

  except RuntimeError as e:
    
    print("Error: " + str(e))

if __name__ == "__main__":
  main()