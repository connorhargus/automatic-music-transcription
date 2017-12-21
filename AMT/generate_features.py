import os
from os import listdir
from os.path import isfile, join

import numpy as np
import math

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

from get_num_mfccs import get_num_mfccs

import pickle

tdnn_inputs = 30


"""
For each audio file use python_speech_features library to extract MFCCs and save them
"""
def generate_features(src_dir, out_file):

    num_files = len(listdir(src_dir)) - 1
    print("Number of files: " + str(num_files))

    feat_vals = []

    # Iterate over audio files
    for i, wav_file in enumerate(sorted(listdir(src_dir))):

        # if i > 2:
        #     break

        if not ".wav" in wav_file:
            continue

        print(wav_file)

        num_mfccs = get_num_mfccs(join(src_dir, wav_file[:-4]))

        (rate,sig) = wav.read(join(src_dir, wav_file))

        print("Sample rate: " + str(rate))

        # Defaults to winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None
        mfcc_feat = mfcc(sig,rate, nfft=1500, winstep=0.01, numcep=13)

        print('mfcc shape: ' + str(mfcc_feat.shape) + '\n')
        # print('mfcc feats: ' + str(mfcc_feat))

        # Alternatively, use delta or filter bank features
        # d_mfcc_feat = delta(mfcc_feat, 2)
        # fbank_feat = logfbank(sig,rate)

        # Make feature vals span windows of MFCCs of size tdnn_inputs
        for set_inputs_i in range(num_mfccs - tdnn_inputs):

            set_inputs = mfcc_feat[set_inputs_i:set_inputs_i+tdnn_inputs, :]
            feat_vals.append(set_inputs)

    # Convert feat_vals from list to numpy array
    feat_vals = np.array(feat_vals, dtype=np.float32)

    # Roll the cepstral co-efficients axis forward for compatibility with Keras
    feat_vals = np.rollaxis(feat_vals, 2, 1)

    print("\nfeat_vals shape: " + str(feat_vals.shape))
    print("Writing features file " + str(out_file) + "...")

    pickle.dump(feat_vals, open(out_file, 'wb'))


if __name__ == '__main__':

    print("Running generate_features...")

    # src_dir = 'audio_files/train'
    # out_file = 'data/tdnn/mfcc_feat_train.pkl'

    # src_dir = 'audio_files/test'
    # out_file = 'data/tdnn/mfcc_feat_test.pkl'

    # generate_features(src_dir, out_file)