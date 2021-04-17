import dataloader
import librosa
import numpy as np

tr_data, tr_targets, tr_sessions = dataloader.load_data(dataloader.TRAIN_DIR, dataloader.REC)
ev_data, ev_targets, ev_sessions = dataloader.load_data(dataloader.DEV_DIR, dataloader.REC)
ts_data = dataloader.load_data(dataloader.EVAL_DIR, dataloader.REC, False)


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


tr_features = []
i = 0
for data in tr_data:
    i += 1
    print(i, "/", len(tr_data))
    tr_features.append(extract_features(data))

x_train = np.array(tr_features)
y_train = np.array(tr_targets)



#import keras
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.callbacks import EarlyStopping
# Build a simple dense model with early stopping and softmax for categorical classification, remember we have 30 classes


#class Sequential(keras.models.Sequential):
    #def __init__(self):
    #    super(Sequential, self).__init__()
    #    self.add(Dense(193, input_shape=(193,), activation = 'relu'))
   #     self.add(Dropout(0.1))
  #      self.add(Dense(128, activation = 'relu'))
 #       self.add(Dropout(0.25))
 #       self.add(Dense(128, activation = 'relu'))
 #       self.add(Dropout(0.5))
 #       self.add(Dense(30, activation = 'softmax'))
#        self.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')        



#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')