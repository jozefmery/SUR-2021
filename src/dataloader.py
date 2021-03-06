# File:     dataloader.py
# Date:     13.05.2021
# Project:  SUR-2021
# Authors:  Dovičic Denis     - xdovic01@vutbr.cz
#           Hudecová Patrícia - xhudec30@vutbr.cz
#           Méry Jozef        - xmeryj00@vutbr.cz
# Description: Dataset loading utilities.    

from enum import Enum
import os
import glob
import numpy as np
import PIL.Image
import scipy.io.wavfile
from utils import identity, concat

class Category(Enum):
  # values represent the directory name
  TRAIN = "train"
  DEV   = "dev"
  EVAL  = "eval"

class System(Enum):
  # values match the positional argument, see main.py - parse_arguments
  FACE  = "face"
  VOICE = "voice"

_SYS_TO_FILE_EXT = {
  System.FACE:  ".png",
  System.VOICE: ".wav"
}

def _load_image(path: str):
  # resize images to a known fixed size
  return np.asarray(PIL.Image.open(path).resize((160, 160)))

def _load_sound(path: str):
  # read audio and convert its data to a numpy array
  rate, audio = scipy.io.wavfile.read(path)
  return np.asarray(audio), rate

def _find_files(path: str, system: System):
  # create the pattern using the system type mapped to a file extension
  return glob.glob(os.path.join(path, "*" + _SYS_TO_FILE_EXT[system]))

_TRANSFORM_FN_MAPPER = {
  System.FACE:  _load_image,
  System.VOICE: _load_sound
}

def _load_files(paths: "list[str]", system: System):
  return [_TRANSFORM_FN_MAPPER[system](p) for p in paths]

def degroup_data(data_and_targets):
  # extract tuple members
  grouped_data = data_and_targets[0]
  group_targets = data_and_targets[1]
  # create target for every sample
  grouped_targets = [[target] * len(group) for group, target in zip(grouped_data, group_targets)]
  # flatten groups
  return concat(grouped_data), concat(grouped_targets)

_SYS_GROUP_MAPPER = {

  System.VOICE: identity,
  System.FACE: degroup_data
}

def _load_dev_train_data(path: str, system: System):
  # discover target directories
  target_dirs = os.listdir(path)
  # find all files
  path_groups = [_find_files(os.path.join(path, target_dir), system) for target_dir in target_dirs]
  # load all files
  grouped_data = [_load_files(group, system) for group in path_groups]
  # create targets
  group_targets = [int(target) for target in target_dirs]
  # degroup the data for the face system
  return _SYS_GROUP_MAPPER[system]((grouped_data, group_targets))

def _load_eval_data(path: str, system: System):
  # create the pattern using the system type mapped to a file extension
  pattern = os.path.join(path, "*" + _SYS_TO_FILE_EXT[system])
  # discover matching file paths
  paths = glob.glob(pattern)
  # load data 
  data = [_TRANSFORM_FN_MAPPER[system](p) for p in paths]
  # extract ids from filenames
  ids = [os.path.basename(p).split(".")[0] for p in paths]
  return data, ids

_LOADER_FN_MAPPER = {
  Category.TRAIN:  _load_dev_train_data,
  Category.DEV:    _load_dev_train_data,
  Category.EVAL:   _load_eval_data, 
}

def load(base_path: str, category: Category, system: System):
  # create path using the Category enum value
  # no need to map enum values
  path = os.path.join(base_path, category.value)
  try:
    return _LOADER_FN_MAPPER[category](path, system)
  except FileNotFoundError as e:
    raise RuntimeError("Failed to load data: " + str(e))