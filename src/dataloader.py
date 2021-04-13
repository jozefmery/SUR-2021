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

def load_rec(path: str) -> np.ndarray:
    return np.asarray(scipy.io.wavfile.read(path)[1])

def load_data(path: str, ext: str, with_target: bool=True) -> tuple:
    classes = glob.glob(path + "*")
    
    data = []
    if with_target:
        targets = []
        sessions = []
        index = 0
    for c in classes:
        cls = {"data": [], "target": [], "session": []}
        for obj in glob.glob(c + "/*." + ext):
            
            if ext == "png":
                cls["data"].append(load_img(obj))
            elif ext == "wav":
                cls["data"].append(load_rec(obj))

            if with_target:
                cls["target"].append(index)
                cls["session"].append(obj[len(c) + 6:len(c) + 8])

        data.append(cls["data"])
        if with_target:
            targets.append(cls["target"])
            sessions.append(cls["session"])
            index += 1

    if with_target:
        return data, targets, sessions
    else:
        return data


if __name__ == "__main__":
    load_data(TRAIN_DIR, REC)
