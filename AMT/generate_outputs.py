import os
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import pickle

# This is Python-Midi package
import midi

# Mido package
from mido import MidiFile
from mido import tick2second

from get_num_mfccs import get_num_mfccs

feature_gap = 0.005
bottom_note = 21
num_notes = 88

# Use the same as in generate_features to trim matrix of outputs
max_song_len = 5000

# Number of input vector MFCCs to the TDNN
tdnn_inputs = 30


"""
Read supplied midi files to extract information on which pitches are turned on at each MFCC sample point
in a recording. An extra dimension is added representing the presence of silence.
"""
def generate_outputs(src_dir, src_audio_dir, out_file):

    num_files = len(listdir(src_dir)) - 1
    print("Number of files: " + str(num_files))

    # List of output arrays of notes turned on for each sample
    output_vals = []

    # For each midi file in the midi source directory, find the notes present in each mfcc window
    for file_i, midi_file in enumerate(sorted(listdir(src_dir))):

        # if file_i > 1:
        #     break

        if ".mid" not in midi_file:
            continue

        # Calling get_num_mfccs here and in the generate_features script ensures our number of features aligns
        # with the number of target outputs
        num_mfccs = get_num_mfccs(join(src_audio_dir, midi_file[:-4]))
        print("Num mfccs: " + str(num_mfccs))
        print("File name: " + str(midi_file))

        # Song outputs temporarily holds all the target outputs for a given recording,
        # before putting those from each recording together into output_vals
        song_outputs = np.zeros((num_mfccs, num_notes + 1))

        pattern = midi.read_midifile(join(src_dir, midi_file))

        # Make MIDI ticks in absolute time instead of relative to previous note
        pattern.make_ticks_abs()

        mid = MidiFile(join(src_dir, midi_file))
        tempo = 0

        # Use mido package to get time information
        for i, track in enumerate(mid.tracks):
            # print('Track {}: {}'.format(i, track.name))

            for msg in track:

                if msg.type == 'set_tempo':
                    tempo = msg.tempo

        # For each note fill in the output array with 1's for the time slot indicating that note is present
        for i in range(0, len(pattern[0])):

            cur_string = str(pattern[0][i])

            if not 'NoteOn' in cur_string:
                continue

            # Get start time and pitch value of note
            start_sec, note = get_note(cur_string, mid.ticks_per_beat, tempo)
            end_sec = None

            # For each note that gets turned on, find the corresponding note off time for that pitch
            for j in range(i+1, len(pattern[0])):

                end_string = str(pattern[0][j])

                if not 'NoteOff' in end_string:
                    continue

                end_sec, note_end = get_note(end_string, mid.ticks_per_beat, tempo)

                if note_end == note:
                    break

            # Round note start and end points to nearest time frame
            start_vec = int(round(float(start_sec)/feature_gap))
            end_vec = int(round(float(end_sec)/feature_gap))
            # print("Note " + str(note - bottom_note) + " present from " + str(start_vec) + " to " + str(end_vec) + " frames")

            # Turn on given note across it's time span in song_outputs matrix
            song_outputs[start_vec:end_vec, note - bottom_note] = 1

        # Append song_output target notes to our list of output_vals across all recordings
        for i, row in enumerate(song_outputs):

            # Train TDNN on note present at the center of the 30 mfcc window
            if tdnn_inputs/2 <= i < num_mfccs - tdnn_inputs/2:
                output_vals.append(row)

    # Convert output_vals from list to numpy array
    output_vals = np.array(output_vals, dtype=np.bool)

    # Find rows of output_vals without any notes present, mark them with a silence "note"
    any_notes = np.sum(output_vals, axis=1)
    silence_vals = np.array(any_notes == 0, dtype=np.bool)
    output_vals[:, num_notes] = silence_vals

    # Save the outputs to a pickle file
    pickle.dump(output_vals, open(out_file, 'wb'))


"""
For a given midi file "note on" line, determine the start time and pitch value of
the note.
"""
def get_note(cur_string, ticks_per_beat, tempo):
    tick_string = ''
    note_string = ''

    # Get the string indices of the tick start and end
    try:
        begin = 'tick='
        end = ', '

        tick_begin = cur_string.find(begin) + len(begin)
        tick_end = cur_string.find(end, tick_begin)
        tick_string = cur_string[tick_begin:tick_end]

        # print('tick_string: ' + str(tick_string))

    except ValueError:
        print('Could not find \"tick=\"')

    # Get the current note pitch value
    try:
        begin = 'data=['
        end = ', '

        note_begin = cur_string.find(begin) + len(begin)
        note_end = cur_string.find(end, note_begin)
        note_string = cur_string[note_begin:note_end]

    except ValueError:
        print('Could not find \"data=[\"')

    # Convert ticks to absolute time using mido library
    ticks = float(tick_string)
    seconds = tick2second(ticks, ticks_per_beat, tempo)

    # Int representing note pitch
    note = int(note_string)

    return seconds, note


if __name__ == '__main__':

    print("Running generate_outputs...")

    # src_dir = 'midi_files/train'
    # src_audio_dir = 'audio_files/train'
    # out_file = 'data/tdnn/target_train.pkl'

    # src_dir = 'midi_files/test'
    # src_audio_dir = 'audio_files/test'
    # out_file = 'data/tdnn/target_test.pkl'
    #
    # generate_outputs(src_dir, src_audio_dir, out_file)

