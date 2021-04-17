import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import dataloader
import librosa
import numpy as np


TRAIN_DIR = dataloader.TRAIN_DIR
DEV_DIR = dataloader.DEV_DIR


def extract_features(data):
    sample_rate, rec = data
    rec = rec.astype(np.float)
    hormonic_rec = librosa.effects.harmonic(rec)

    mfcc = np.mean(librosa.feature.mfcc(y=rec, sr=sample_rate, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(rec))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(rec, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(hormonic_rec, sr=sample_rate).T, axis=0)

    return np.concatenate((mfcc, chroma, mel, contrast, tonnetz), axis=0)


def convert(data_type: int=TRAIN_DIR):
    if data_type == TRAIN_DIR:
        dst = dataloader.TRAIN_VOICE_NPY
    elif data_type == DEV_DIR:
        dst = dataloader.DEV_VOICE_NPY
    else:
        print("Unknow data type - ", data_type, file=sys.stderr)
        return

    i = 0
    data, targets, sessions = dataloader.load_data(data_type, dataloader.REC)
    for x, t, s in zip(data, targets, sessions):
        features_w_target = np.append(extract_features(x), [t])
        i += 1
        np.save(dst + str(t) + "_" + s + "_" + str(i), features_w_target)
        print(i, "/", len(data))


def load_train_data() -> tuple:
    data = dataloader.load_npy(dataloader.TRAIN_VOICE_NPY)
    return data[:, :-1], data[:, -1]


def load_dev_data() -> tuple:
    data = dataloader.load_npy(dataloader.DEV_VOICE_NPY)
    return data[:, :-1], data[:, -1]


if __name__ == "__main__":
    convert(DEV_DIR)
    #load_train_data()
