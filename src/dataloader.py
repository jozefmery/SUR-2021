import os
from PIL import Image
import numpy as np

BASE_DIR = "dataset/"
DEV_DIR = BASE_DIR + "dev/"
EVAL_DIR = BASE_DIR + "eval/"
TRAIN_DIR = BASE_DIR + "train/"


def load_faces(path: str) -> np.ndarray:
    classes = [x[0] for x in os.walk(path)]


    return faces


if __name__ == "__main__":
    load_faces(TRAIN_DIR)

