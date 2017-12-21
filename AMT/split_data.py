import os
from os import listdir
from os.path import isfile, join
import shutil

import numpy as np
np.random.seed(7)

"""
Script to split audio (.wav) and midi files into training and test sets
"""
def split_data(wav_in, wav_out, mid_in, mid_out):

    # Create new train and test folders for audio and midi files
    if os.path.exists(join(wav_out, 'train')):
        shutil.rmtree(join(wav_out, 'train'))
    os.makedirs(join(wav_out, 'train'))

    if os.path.exists(join(wav_out, 'test')):
        shutil.rmtree(join(wav_out, 'test'))
    os.makedirs(join(wav_out, 'test'))

    if os.path.exists(join(mid_out, 'train')):
        shutil.rmtree(join(mid_out, 'train'))
    os.makedirs(join(mid_out, 'train'))

    if os.path.exists(join(mid_out, 'test')):
        shutil.rmtree(join(mid_out, 'test'))
    os.makedirs(join(mid_out, 'test'))

    # Create corpus folder for midi files in corpus
    if os.path.exists(join(mid_out, 'corpus')):
        shutil.rmtree(join(mid_out, 'corpus'))
    os.makedirs(join(mid_out, 'corpus'))

    num_files = len(listdir(wav_in))

    # Use a random permutation to randomly select files fro training and testing
    rand_perm = np.random.permutation(num_files + 1)

    # For each file in the audio folder, find its associated midi file and put them both in
    # either training or testing folders
    for i, wav_file in enumerate(listdir(wav_in)):

        if '.wav' not in wav_file:
            continue

        num_train_recordings = 200
        num_test_recordings = 40

        # Copy midi to test or train midi folder
        mid_file = wav_file[0:-4] + '.mid'

        # Create corpus out of recordings not used in the test set
        if not num_train_recordings <= rand_perm[i] < num_train_recordings + num_test_recordings:
            shutil.copyfile(join(mid_in, mid_file), join(mid_out, 'corpus', mid_file))

        if rand_perm[i] < num_train_recordings or "MAPS" in wav_file:
            dest = 'train'
        elif rand_perm[i] < num_train_recordings + num_test_recordings:
            dest = 'test'
        else:
            continue

        # Copy audio to test or train audio folder
        shutil.copyfile(join(wav_in, wav_file), join(wav_out, dest, wav_file))
        shutil.copyfile(join(mid_in, mid_file), join(mid_out, dest, mid_file))


if __name__ == '__main__':

    wav_in = 'audio_files/clean'
    wav_out = 'audio_files'
    mid_in = 'midi_files/all'
    mid_out = 'midi_files'
    split_data(wav_in, wav_out, mid_in, mid_out)