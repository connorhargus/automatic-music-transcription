from os.path import isfile, join

from split_data import split_data
from generate_features import generate_features
from generate_outputs import generate_outputs
from tdnn import tdnn_train
from tdnn import tdnn_predict
from hmm import hmm_train
from hmm import hmm_predict


"""
Main script recipe to process data, train neural net, train HMM, and decode test data
"""
def main():

    # Locations for audio and midi source files (train and test are subfolders)
    audio_folder = 'audio_files'
    midi_folder = 'midi_files'

    # Subfolder in audio_folder to get wav files from, i.e. 'clean' or 'noise'
    audio_source = 'noise'

    # Locations for input data to tdnn
    tdnn_feat_train = 'data/tdnn/mfcc_feat_train.pkl'
    tdnn_feat_test = 'data/tdnn/mfcc_feat_test.pkl'
    tdnn_target_train = 'data/tdnn/target_train.pkl'
    tdnn_target_test = 'data/tdnn/target_test.pkl'
    target_corpus = 'data/hmm/target_corpus'

    # Locations for output data from tdnn
    tdnn_probs_train = 'data/hmm/train_probs.pkl'
    tdnn_probs_test = 'data/hmm/test_probs.pkl'
    notes_train = 'data/hmm/train_notes.pkl'
    notes_test = 'data/hmm/test_notes.pkl'

    # Path for saved tdnn model and hmm model
    tdnn_model_name = "models/tdnn.h5"
    hmm_model_name = "models/hmm.pkl"

    # Path for resulting midi files from tdnn and hmm
    output_midi_tdnn = 'output_midi/output_midi_tdnn.mid'
    output_midi_hmm = 'output_midi/output_midi_hmm.mid'

    stage = 1

    # Split data into train and test sets
    if stage <= 1:

        print("\nSplitting data...\n")

        wav_in = join(audio_folder, audio_source)
        wav_out = audio_folder
        mid_in = join(midi_folder, 'all')
        mid_out = midi_folder

        split_data(wav_in, wav_out, mid_in, mid_out)

        print("\nFinished splitting data...\n")

    # Create MFCC features for each test and train audio file
    if stage <= 2:

        print("\nGenerating MFCCs...\n")

        src_dir = 'audio_files/train'
        generate_features(src_dir, tdnn_feat_train)

        src_dir = 'audio_files/test'
        generate_features(src_dir, tdnn_feat_test)

        print("\nFinished generating MFCCs...\n")

    # Generate expected pitches for each MFCC from test, train, and corpus midi files
    if stage <= 3:

        print("\nGenerating target pitches...\n")

        src_dir = 'midi_files/train'
        src_audio_dir = 'audio_files/train'
        generate_outputs(src_dir, src_audio_dir, tdnn_target_train)

        src_dir = 'midi_files/test'
        src_audio_dir = 'audio_files/test'
        generate_outputs(src_dir, src_audio_dir, tdnn_target_test)

        # Generate expected pitches for corpus files (we don't need MFCCs for LM)
        src_dir = 'midi_files/corpus'
        src_audio_dir = 'audio_files/clean'
        generate_outputs(src_dir, src_audio_dir, target_corpus)

        print("\nFinished generating target pitches...\n")

    # Train TDNN on MFCCs and target pitch data
    if stage <= 4:

        print("\nTraining time-delay neural network...\n")

        tdnn_train(tdnn_feat_train, tdnn_target_train, tdnn_model_name)

        print("\nFinished training time-delay neural network...\n")

    # Make predictions based on TDNN
    if stage <= 5:

        print("\nPredicting note probabilities using TDNN...\n")

        tdnn_predict(tdnn_model_name, tdnn_feat_train, tdnn_target_train, tdnn_feat_test, tdnn_target_test, output_midi_tdnn)

        print("\nFinished predicting note probabilities using TDNN...\n")

    # Train HMM on TDNN output probabilities
    if stage <= 6:

        print("\nTraining hidden markov model...\n")

        hmm_train(tdnn_probs_train, notes_train, target_corpus, tdnn_target_train, hmm_model_name)

        print("\nFinished training hidden markov model...\n")

    # Train HMM on TDNN output probabilities
    if stage <= 7:

        print("\nDecoding hidden markov model...\n")

        hmm_predict(tdnn_probs_test, notes_test, hmm_model_name, output_midi_hmm)

        print("\nFinished decoding hidden markov model...\n")

    # # Compute the per-frame accuracy and WER scores for the TDNN and HMM
    # if stage <= 8:



if __name__ == '__main__':

    main()