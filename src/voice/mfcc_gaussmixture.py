import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile
import glob
import os
import PIL
from sklearn import preprocessing
import python_speech_features as mfcc

BASE_DIR = "dataset\\"
DEV_DIR = BASE_DIR + "dev\\"
TEST_DIR = BASE_DIR + "eval\\"
TRAIN_DIR = BASE_DIR + "train\\"

REC = "wav"
FACE = "png"

MODEL_DIR = "model\\voice\\mfcc_gauss"

def load_img(path: str) -> np.ndarray:
    return np.asarray(PIL.Image.open(path).resize((160, 160)))

def load_rec(path: str):
    return scipy.io.wavfile.read(path)

def load_data(path: str, ext: str):
    data = []
    targets = []
    for cls, target in zip(glob.glob(path + "*"), os.listdir(path)):
        target_data = []
        for obj in glob.glob(cls + "/*." + ext):

            if ext == "png":
                target_data.append(load_img(obj))
            elif ext == "wav":
                target_data.append(load_rec(obj))

        targets.append(target)
        data.append(target_data)

    return data, targets


def load_models(path: str):
    
    targets = os.listdir(path)
    models = [pickle.load(open(glob.glob(path + "/*.gmm")[0], 'rb')) for path in glob.glob(path + "\\*")]

    return models, targets


def extract_features(audio, rate):    
    mfcc_feat = mfcc.mfcc(audio, rate, numcep=96, nfilt=96, nfft=1024)
    return preprocessing.scale(mfcc_feat)


def train():
    data, targets = load_data(TRAIN_DIR, REC)

    for audios, target in zip(data, targets):
        features = np.asarray(())
        for sample_rate, audio in audios:
            feature = extract_features(audio, sample_rate)

            if features.size == 0:
                features = feature
            else:
                features = np.vstack((features, feature))
        
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='tied', n_init=3)
        gmm.fit(features)
 
        os.makedirs(os.path.join(MODEL_DIR, target), exist_ok=True)
        pickle.dump(gmm, open(os.path.join(MODEL_DIR, target, "speaker_" + target + ".gmm"), 'wb'))
        print('+ modeling completed for speaker:', target)


def eval():
    data, targets = load_data(DEV_DIR, REC)
    models, clss = load_models(MODEL_DIR)

    identified = 0
    for target, audios in zip(targets, data):   
        for sample_rate, audio in audios:
            vector = extract_features(audio, sample_rate)
            
            log_likelihood = np.zeros(len(models))
            for i in range(len(models)):
                scores = np.array(models[i].score(vector))
                log_likelihood[i] = scores.sum()
            
            winner = clss[np.argmax(log_likelihood)]
            print(winner, "==", target)
            if winner == target:
                identified += 1

    print(identified, "/", len(data))
    print(identified / float(len(data)))
    

def test():
    models, clss = load_models(MODEL_DIR)

    data = np.ndarray((0, 33))
    i = 0
    for path, record in zip(glob.glob(TEST_DIR + "*"), os.listdir(TEST_DIR)):
        i += 1
        if i > 50:
            break
        if record.endswith(".wav"):
            sample_rate, audio = scipy.io.wavfile.read(path)
            vector = extract_features(audio, sample_rate)

            probs = []
            log_likelihood = np.zeros(len(models))
            for i in range(len(models)):
                scores = np.array(models[i].score(vector))
                log_likelihood[i] = scores.sum()
                probs.append("NaN")
            winner = clss[np.argmax(log_likelihood)]

            row = np.concatenate(([record[:-4], winner], probs))
            data = np.append(data, [row], axis=0)

    np.savetxt("evaluation.txt", data, fmt="%s")
    

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "eval":
        eval()
    elif sys.argv[1] == "test":
        test()
    