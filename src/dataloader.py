import os
import glob
import PIL.Image
import scipy.io.wavfile
import numpy as np


BASE_DIR = "dataset/"
DEV_DIR = BASE_DIR + "dev/"
EVAL_DIR = BASE_DIR + "eval/"
TRAIN_DIR = BASE_DIR + "train/"

REC = "wav"
FACE = "png"

SIGNAL_TRAIN_VOICE = "data/voice/signal/train/"
SIGNAL_DEV_VOICE = "data/voice/signal/dev/"

SPECTOG_TRAIN_VOICE = "data/voice/spectogram/train/"
SPECTOG_DEV_VOICE = "data/voice/spectogram/dev/"


def load_img(path: str) -> np.ndarray:
    return np.asarray(PIL.Image.open(path).resize((160, 160)))


def load_rec(path: str):
    return scipy.io.wavfile.read(path)


def load_data(path: str, ext: str, with_target: bool=True):
    classes = glob.glob(path + "*")
    
    data = []
    if with_target:
        targets = []
        sessions = []
    for c, t in zip(classes, os.listdir(path)):
        for obj in glob.glob(c + "/*." + ext):

            if ext == "png":
                data.append(load_img(obj))
            elif ext == "wav":
                data.append(load_rec(obj))

            if with_target:
                targets.append(t)
                sessions.append(obj[len(c) + 6:len(c) + 8])

    if with_target:
        return data, targets, sessions
    else:
        return data


def load_npy(path: str) -> np.ndarray:
    data = []
    for file in glob.glob(path + "*"):
        data.append(np.load(file))

    return np.asarray(data)

def load_spectog(path: str) -> np.ndarray:
    data = []
    for file in glob.glob(path + "*"):
        data.append(np.load(file, allow_pickle=True))

    return np.asarray(data)


if __name__ == "__main__":
    #load_spectog(SPECTOG_DEV_VOICE)
    load_data(TRAIN_DIR, ext=REC)
    #load_npy(TRAIN_VOICE_NPY)
