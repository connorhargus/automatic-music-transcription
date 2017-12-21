#!/bin/bash

# Convert .wav files to .sph for use by the Tedlium example, 16000kHz sampling rate with 1 channel

for i in wav_files/*.wav
do
sox -t wav -c 1 $i -t sph -c 1 -r 16000 --endian big sph_files/$(basename $i .wav).sph
done