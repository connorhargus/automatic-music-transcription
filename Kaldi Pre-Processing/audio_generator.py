import os
from os import listdir
from os.path import isfile, join

from midi2audio import FluidSynth


"""
Generates a .wav audio file for each midi file in src_dir and places
them in out_dir. Uses FluidSynth to perform the audio synthesis.

Note: This file must be run before pre_process.py in order to have the audio
files, but it requires FluidSynth which can be a pain to install so I have provided
audio files created by this script in my submission.
"""
def generate_audio(src_dir, sf2_path, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, file in enumerate(sorted(listdir(src_dir))):

        new_file = 'saxophone_' + str(i)

        if file == '.DS_Store':
            continue

        print(str(file))

        fs = FluidSynth(sample_rate=16000)

        # To pass in other sf2 soundfont:
        # FluidSynth(sf2_path)
        # FluidSynth().play_midi(join(src_dir, file))

        fs.midi_to_audio(join(src_dir, file), join(out_dir, new_file + '.wav'))


if __name__ == '__main__':

    sf2_path = 'soundfonts/JR_sax.sf2'
    src_dir = 'midi_files'
    out_dir = 'wav_files'

    generate_audio(src_dir, sf2_path, out_dir)

