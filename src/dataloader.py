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


def load_img(path: str) -> np.ndarray:
    return np.asarray(PIL.Image.open(path).resize((160, 160)))

def load_rec(path: str):
    return scipy.io.wavfile.read(path)

def load_data(path: str, ext: str, with_target: bool=True) -> tuple:
    classes = glob.glob(path + "*")
    
    data = []
    if with_target:
        targets = []
        sessions = []
        index = 0
    for c in classes:
        for obj in glob.glob(c + "/*." + ext):
            
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


if __name__ == "__main__":
    load_data(TRAIN_DIR, REC)
