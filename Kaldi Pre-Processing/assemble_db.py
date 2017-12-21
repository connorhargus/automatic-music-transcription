import os
from os import listdir
from os.path import isfile, join

import numpy as np
np.random.seed(0)
import shutil


"""
Takes midi and generated audio files and puts them in the db/ folder with
the same structure as used in the Tedlium example, splitting the dataset into
train, test, and dev sets in the process.
"""
def assemble_db(stm_dir, sph_dir, out_dir):

    num_files = len(listdir(stm_dir))
    rand_perm = np.random.permutation(num_files + 1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(join(out_dir, 'train', 'stm'))
        os.makedirs(join(out_dir, 'train', 'sph'))
        os.makedirs(join(out_dir, 'test', 'stm'))
        os.makedirs(join(out_dir, 'test', 'sph'))
        os.makedirs(join(out_dir, 'dev', 'stm'))
        os.makedirs(join(out_dir, 'dev', 'sph'))
        os.makedirs(join(out_dir, 'LM'))

    for i, stm_file in enumerate(listdir(stm_dir)):

        if stm_file == '.DS_Store':
            continue

        if rand_perm[i] < 20:
            dest = 'test'
        elif rand_perm[i] < 40:
            dest = 'dev'
        else:
            dest = 'train'

        shutil.copyfile(join(stm_dir, stm_file), join(out_dir, dest, 'stm', stm_file))

        sph_file_name = stm_file[0:-4] + '.sph'
        shutil.copyfile(join(sph_dir, sph_file_name), join(out_dir, dest, 'sph', sph_file_name))


if __name__ == '__main__':

    stm_dir = 'stm_files'
    sph_dir = 'audio_files/audio_sph'
    out_dir = 'db/TEDLIUM_release2'

    assemble_db(stm_dir, sph_dir, out_dir)

