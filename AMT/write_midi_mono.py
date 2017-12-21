from mido import Message, MidiFile, MidiTrack
from mido import tick2second
from mido import second2tick

default_tempo = 500000 # See https://mido.readthedocs.io/en/latest/midi_files.html
bottom_note = 21


"""
Writes a list of notes present at each time frame to a midi file. This allows us
to reconstruct the original midi file after labeling each frame with a note (or silence)
"""
def write_midi_mono(mono_notes, midi_file_name):

    midi_file = MidiFile(ticks_per_beat=10000)

    track = MidiTrack()
    midi_file.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))

    cur_time = 0
    prev_note = None
    for i, note in enumerate(mono_notes):

        cur_time += 0.005

        if note == prev_note:
            continue

        else:
            cur_ticks = second2tick(cur_time, midi_file.ticks_per_beat, default_tempo)

            # If silence, don't write note on and off messages to midi
            if prev_note != 88 and prev_note is not None:
                track.append(Message('note_off', note=int(bottom_note + prev_note), velocity=127, time=int(cur_ticks)))
                cur_time = 0
                cur_ticks = 0

            if note != 88 and i != len(mono_notes) - 1:
                track.append(Message('note_on', note=int(bottom_note + note), velocity=127, time=int(cur_ticks)))
                cur_time = 0

            prev_note = note

    midi_file.save(midi_file_name)
