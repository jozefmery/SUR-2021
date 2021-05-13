import python_speech_features
from sklearn import preprocessing
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from dataloader import degroup_data

def _extract_features(audio, rate):
  # convert audio sample to a feature vector   
  mfcc_feat = python_speech_features.mfcc(audio, rate, numcep=96, nfilt=96, nfft=1024)
  return preprocessing.scale(mfcc_feat)

def _audio_group_to_feature_matrix(group):
  # convert a speaker's recordings to a feature matrix
  return np.vstack([_extract_features(audio, sample_rate) for audio, sample_rate in group])

def _audio_groups_to_feature_matrices(groups):
  # convert every speaker's recordings to individual matrices
  return [_audio_group_to_feature_matrix(group) for group in groups]

def _feature_matrix_to_gmm(features, speaker):
  # create and fit a GMM to the feature matrix
  gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type="tied", n_init=3)
  gmm.fit(features)
  # report progress as training is lengthier 
  print("Finished training GMM for speaker {}".format(speaker))
  return gmm

def train_gmms(train_data, dev_data):
  # convert audio groups to feature matrices
  feature_matrices = _audio_groups_to_feature_matrices(train_data[0])
  # train a gmm for every matrix (speaker)
  gmms = [_feature_matrix_to_gmm(matrix, speaker) for matrix, speaker in zip(feature_matrices, train_data[1])]
  # bundle gmms with the targets
  model = gmms, train_data[1]
  # evaluate the trained model using the dev data
  # remove groups 
  data, targets = degroup_data(dev_data)
  prediction = predict(model, data)
  score = accuracy_score(targets, prediction)
  return model, score

def _predict_sample(model, recording, sample_rate):
  gmms, targets = model
  # extract features from the recording for the prediction
  features = _extract_features(recording, sample_rate)
  # get score from every gmm
  log_likelihood = [np.array(gmm.score(features)).sum() for gmm in gmms]
  # pick winner gmm
  return targets[np.argmax(log_likelihood)]

def predict(model, recordings):
  # make prediction for every sample
  return [_predict_sample(model, recording, sample_rate) for recording, sample_rate in recordings]