import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import dataloader
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn import preprocessing
import python_speech_features as mfcc


TRAIN_DIR = dataloader.TRAIN_DIR
DEV_DIR = dataloader.DEV_DIR





def spectogram_from_signal(data, filename, target):
    sample_rate, rec = data
    plt.interactive(False)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=rec.astype('float32'), sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig(filename + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()


def create_spectrograms(data_type: str=TRAIN_DIR):
    if data_type == TRAIN_DIR:
        dst = dataloader.SPECTOG_TRAIN_VOICE
    elif data_type == DEV_DIR:
        dst = dataloader.SPECTOG_DEV_VOICE
    else:
        print("Unknow data type - ", data_type, file=sys.stderr)
        return

    i = 0
    data, targets, sessions = dataloader.load_data(data_type, dataloader.REC)
    for x, t, s in zip(data, targets, sessions):
        i += 1
        os.makedirs(dst + t, exist_ok=True)
        file = dst + t + "/" + str(t) + "_" + s + "_" + str(i)
        spectogram_from_signal(x, file, t)
        print(i, "/", len(targets))
    

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


def convert_signal_to_npy(data_type: str=TRAIN_DIR) -> None:
    if data_type == TRAIN_DIR:
        dst = dataloader.SIGNAL_TRAIN_VOICE
    elif data_type == DEV_DIR:
        dst = dataloader.SIGNAL_DEV_VOICE
    else:
        print("Unknow data type - ", data_type, file=sys.stderr)
        return

    i = 0
    data, targets, sessions = dataloader.load_data(data_type, dataloader.REC)
    for x, t, s in zip(data, targets, sessions):
        features_w_target = np.append(extract_features(x), [t])
        i += 1
        np.save(dst + str(t) + "_" + s + "_" + str(i), features_w_target)
        print(i, "/", len(targets))


def load_spectogram_train_data() -> tuple:
    data = dataloader.load_spectog(dataloader.SPECTOG_TRAIN_VOICE)
    return np.asarray(data[:, :-1][:, 0]), data[:, -1]


def load_spectogram_dev_data() -> tuple:
    data = dataloader.load_spectog(dataloader.SPECTOG_DEV_VOICE)
    return np.asarray(data[:, :-1][:, 0]), data[:, -1]


def load_signal_train_data() -> tuple:
    data = dataloader.load_npy(dataloader.SIGNAL_TRAIN_VOICE)
    return data[:, :-1], data[:, -1]


def load_signal_dev_data() -> tuple:
    data = dataloader.load_npy(dataloader.SIGNAL_DEV_VOICE)
    return data[:, :-1], data[:, -1]


if __name__ == "__main__":
    create_spectrograms(_DIR)
    #convert_to_npy(DEV_DIR)
    #load_train_data()
