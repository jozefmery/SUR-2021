import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile
import glob
import os
import utils
from utils import load_data, extract_features, load_models


def train():
    data, targets = load_data(utils.TRAIN_DIR, utils.REC)

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
 
        os.makedirs(os.path.join(utils.MODEL_DIR, target), exist_ok=True)
        pickle.dump(gmm, open(os.path.join(utils.MODEL_DIR, target, "speaker_" + target + ".gmm"), 'wb'))
        print('+ modeling completed for speaker:', target)


def eval():
    data, targets = load_data(utils.DEV_DIR, utils.REC)
    models, clss = load_models(utils.MODEL_DIR)

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
    models, clss = load_models(utils.MODEL_DIR)

    data = np.ndarray((0, 33))
    i = 0
    for path, record in zip(glob.glob(utils.TEST_DIR + "*"), os.listdir(utils.TEST_DIR)):
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
    