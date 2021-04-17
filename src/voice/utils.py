import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import dataloader
import librosa
import numpy as np

#tr_data, tr_targets, tr_sessions = dataloader.load_data(dataloader.TRAIN_DIR, dataloader.REC)
#ev_data, ev_targets, ev_sessions = dataloader.load_data(dataloader.DEV_DIR, dataloader.REC)
#ts_data = dataloader.load_data(dataloader.EVAL_DIR, dataloader.REC, False)


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


def convert(data_type: int=dataloader.TRAIN_DIR):
    data, targets, sessions = dataloader.load_data(data_type, dataloader.REC)

    i = 0
    for x, t, s in zip(data, targets, sessions):
        features_w_target = np.append(extract_features(x), [t])
        i += 1
        np.save("data/voice/" + str(t) + "_" + s + "_" + str(i), features_w_target)
        print(i, "/", len(data))




convert(dataloader.TRAIN_DIR)