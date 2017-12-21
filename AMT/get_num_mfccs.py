import os
from os import listdir
from os.path import isfile, join
import math

# Use wav and contextlib to get recording duration
import wave
import contextlib


"""
Use the sample rate of an audio file to predict the number of MFCCs which will be generated
"""
def get_num_mfccs(file):
    fname = join(file + ".wav")
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    num_mfccs = int(math.floor(duration / 0.005))
    print("Length of song: " + str(duration) + " seconds")

    # Return conservative number of mfccs (subtract 2) to make sure we have enough
    return num_mfccs - 2