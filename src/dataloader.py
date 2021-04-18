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

TRAIN_VOICE_NPY = "data/voice/train/"
DEV_VOICE_NPY = "data/voice/dev/"


def load_img(path: str) -> np.ndarray:
    return np.asarray(PIL.Image.open(path).resize((160, 160)))


def load_rec(path: str):
    return scipy.io.wavfile.read(path)


def load_data(path: str, ext: str, with_target: bool=True) -> tuple:
    classes = glob.glob(path + "*")
    classes2 = glob.glob(DEV_DIR + "*")
    
    data = []
    if with_target:
        targets = []
        sessions = []
        index = 0
    for c, c2 in zip(classes, classes2):
        for obj, obj2 in zip(glob.glob(c + "/*." + ext), glob.glob(c2 + "/*.wav")):
            
            print(obj, "==", obj2)

            if ext == "png":
                data.append(load_img(obj))
            elif ext == "wav":
                data.append(load_rec(obj))

            if with_target:
                targets.append(index)
                sessions.append(obj[len(c) + 6:len(c) + 8])

        if with_target:
            index += 1

    if with_target:
        return data, targets, sessions
    else:
        return data


def load_npy(path: str) -> np.ndarray:
    data = []
    for file in glob.glob(path + "*"):
        data.append(np.load(file))

    return np.asarray(data)


if __name__ == "__main__":
    load_data(TRAIN_DIR, REC)
    #load_npy(TRAIN_VOICE_NPY)
