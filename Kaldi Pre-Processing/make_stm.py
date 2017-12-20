import os
from os import listdir
from os.path import isfile, join

# This is Python-Midi package
import midi

# Mido package
from mido import MidiFile
from mido import tick2second

target_utt_length = 2

def main(src_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(join(out_dir))

    # For each midi file in the midi source directory, generate utterances by grouping notes
    for i, midi_file in enumerate(sorted(listdir(src_dir))):

        # if i > 1:
        #     return

        if midi_file == '.DS_Store':
            continue

        pattern = midi.read_midifile(join(src_dir, midi_file))

        # Make MIDI ticks in absolute time instead of relative to previous note
        pattern.make_ticks_abs()

        # print(len(pattern[0]))
        # print(pattern)

        mid = MidiFile(join(src_dir, midi_file))

        new_file = 'saxophone_' + str(i)

        with open(join(out_dir, new_file + '.stm'), 'w') as stm_file:

            tempo = 0

            num_off = 0
            num_on = 0

            # Use mido package to get time information
            for i, track in enumerate(mid.tracks):
                # print('Track {}: {}'.format(i, track.name))

                for msg in track:

                    print(msg)

                    if msg.type == 'set_tempo':
                        tempo = msg.tempo

                    elif msg.type == 'note_off':
                        num_off = num_off +  1

                    elif msg.type == 'utterance_on':
                        num_on = num_on + 1


            utterance_on = False

            utt_notes = []

            # Time at which utterance began
            began_utterance = 0

            # Time at which last note ended, used in order to get silence at beginning of utterances
            last_note_end_time = 0

            # Iteratively go through notes and record strings of notes close together as utterances
            for i in range(0, len(pattern[0])):

                cur_string = str(pattern[0][i])
                print(cur_string)

                if 'NoteOn' in cur_string:
                    note_on = True
                elif 'NoteOff' in cur_string:
                    note_on = False
                else:
                    continue

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
                seconds = tick2second(ticks, mid.ticks_per_beat, tempo)
                # print('seconds: ' + str(seconds))

                # Int representing note pitch
                note = int(note_string)

                # If hasn't started an utterance, start one with the current note
                if not utterance_on and note_on:

                    seconds = max(last_note_end_time + 0.03, seconds - 0.2)

                    stm_file.write(new_file + ' 1 ' + new_file + ' ' + "{:.2f}".format(seconds) + ' ')
                    utterance_on = True

                    utt_notes.append(note)

                    began_utterance = seconds

                # If new note is within target_utt_length seconds of the first note in the utterance, include the new
                # note withing the utterance by appending it to utt_notes
                elif seconds <= began_utterance + target_utt_length and note_on:

                    utt_notes.append(note)

                # If the target_utt_length has passed, finish the utterance string by appending the end time and the
                # list of notes in the utterance
                elif not note_on and seconds > began_utterance + target_utt_length:
                    stm_file.write("{:.2f}".format(seconds) + ' <o,f0,unknown> ' + ' '.join(map(str, utt_notes)) + '\n')
                    utterance_on = False

                    utt_notes = []

                    last_note_end_time = seconds

            # # If a note is still on at the end, cut it off
            # if utterance_on:
            #     stm_file.write("{:.2f}".format(last_on_time + 0.01) + ' <o,f0,unknown> ' + str(last_on_note) + '\n')


        # Make sure there are no utterances lasting a very short amount of time
        with open(join(out_dir, new_file + '.stm'), 'r') as stm_file:

            with open(join(out_dir, new_file + '.stm.tmp'), 'w') as tmp_stm_file:

                for line in stm_file.readlines():

                    if len(line.split()) <= 4:
                        continue

                    print(line)
                    if float(line.split()[3]) < float(line.split()[4]) - 1:

                        tmp_stm_file.write(line)

        os.rename(join(out_dir, new_file + '.stm.tmp'), join(out_dir, new_file + '.stm'))


if __name__ == '__main__':

    src_dir = 'midi_files'
    out_dir = 'stm_files'

    main(src_dir, out_dir)

