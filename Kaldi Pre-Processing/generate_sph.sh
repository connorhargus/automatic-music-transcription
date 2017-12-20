#!/bin/bash

for i in audio_files/*.wav
do
sox -t wav -c 1 $i -t sph -c 1 -r 16000 --endian big audio_files/audio_sph/$(basename $i .wav).sph
done