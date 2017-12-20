import os
from os import listdir
from os.path import isfile, join

from midi2audio import FluidSynth

def main(src_dir, sf2_path, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(join(out_dir, 'audio_sph'))

    for i, file in enumerate(sorted(listdir(src_dir))):

        new_file = file[:-4]

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
    src_dir = 'midi_files/all'
    # out_dir = 'audio_files'
    out_dir = 'audio_files/clean'

    main(src_dir, sf2_path, out_dir)

