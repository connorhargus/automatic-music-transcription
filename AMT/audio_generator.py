import os
from os import listdir
from os.path import isfile, join

from midi2audio import FluidSynth


"""
Generates a .wav audio file for each midi file in src_dir and places
them in out_dir. Uses FluidSynth to perform the audio synthesis. Note:
this differs slightly from the audio_generator.py present in the Kaldi
Pre-Processing folder. Importantly, it preserves file names for easier
look up in debugging.
"""
def main(src_dir, sf2_path, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, file in enumerate(sorted(listdir(src_dir))):

        new_file = file[:-4]

        if file == '.DS_Store':
            continue

        print(str(file))

        fs = FluidSynth(sample_rate=16000)

        fs.midi_to_audio(join(src_dir, file), join(out_dir, new_file + '.wav'))

if __name__ == '__main__':

    sf2_path = 'soundfonts/JR_sax.sf2'
    src_dir = 'midi_files/all'
    out_dir = 'audio_files/clean'

    main(src_dir, sf2_path, out_dir)

