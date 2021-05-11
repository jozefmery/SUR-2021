import scipy.io.wavfile
import glob
import os
import pickle
import python_speech_features
from sklearn import preprocessing


BASE_DIR = "dataset\\"
DEV_DIR = BASE_DIR + "dev\\"
TEST_DIR = BASE_DIR + "eval\\"
TRAIN_DIR = BASE_DIR + "train\\"

REC = "wav"

MODEL_DIR = "model\\voice\\mfcc_gauss"


def load_rec(path: str):
    return scipy.io.wavfile.read(path)

def load_data(path: str, ext: str):
    data = []
    targets = []
    for cls, target in zip(glob.glob(path + "*"), os.listdir(path)):
        target_data = []
        for obj in glob.glob(cls + "/*." + ext):

            if ext == "wav":
                target_data.append(load_rec(obj))

        targets.append(target)
        data.append(target_data)

    return data, targets


def load_models(path: str):
    
    targets = os.listdir(path)
    models = [pickle.load(open(glob.glob(path + "/*.gmm")[0], "rb")) for path in glob.glob(path + "\\*")]

    return models, targets


def extract_features(audio, rate):    
    mfcc_feat = python_speech_features.mfcc(audio, rate, numcep=96, nfilt=96, nfft=1024)
    return preprocessing.scale(mfcc_feat)