import os
import logging as log
import pickle
from pathlib import Path

# call before importing tf or keras
def set_tf_loglevel_warn():
  set_tf_loglevel(log.WARNING)

def set_tf_loglevel(level):

  if level >= log.FATAL:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  elif level >= log.ERROR:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  elif level >= log.WARNING:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  log.getLogger("tensorflow").setLevel(level)

def enable_xla_devices():
  # https://www.tensorflow.org/xla
  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

def mkdir(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def load_model(path):
  return pickle.load(open(path, "rb"))

def save_model(path, model):
  # create target directory if needed
  mkdir(os.path.dirname(path))
  pickle.dump(model, open(path, "wb"))

def NaNs_string(n):
  return " ".join(["NaN"] * n)

def write_results(path, ids, predictions):
  mkdir(os.path.dirname(path))
  results = "\n".join(["{} {} {}".format(ident, pred, NaNs_string(31)) for ident, pred in zip(ids, predictions)])

  with open(path, "w") as f:
    f.write(results)

def identity(arg, *args):
  return (arg,) + args if args else arg