import pickle
import numpy as np
from itertools import groupby

from hmmlearn import hmm

from write_midi_mono import write_midi_mono


NUM_NOTES = 89 # 88 + 1 for silence


"""
Train a hidden markov model in hmmlearn to capture transitions between notes 
from frame to frame. Training is performed using maximum likelihood estimate since
hidden states (notes) are observed in the training data.
"""
def hmm_train(tdnn_probs_train, target_notes_train, target_corpus, notes_train, output_model_file):

    X_train = pickle.load(open(tdnn_probs_train, 'rb'))
    # Note: here Y_train is list of notes in number form rather than one-hot encoding
    Y_train = pickle.load(open(target_notes_train, 'rb'))

    # Convert target_corpus to list of desired notes
    target_corpus = pickle.load(open(target_corpus, 'rb'))
    target_corpus = np.argmax(target_corpus, axis=1)

    print('X_train (input probabilities) shape: ' + str(X_train.shape))
    print('Y_train (notes) shape: ' + str(Y_train.shape))
    print('Corpus (notes) shape: ' + str(target_corpus.shape))

    note_hmm = hmm.GaussianHMM(n_components=NUM_NOTES, covariance_type="full", n_iter=100)

    count_transitions = np.zeros((NUM_NOTES, NUM_NOTES))

    prev_note = 88
    for note in target_corpus:
        count_transitions[prev_note, note] += 1
        prev_note = note

    # Simple plus 0.01 smoothing
    count_transitions = count_transitions + 0.01
    # print('Sample of count_transitions: \n' + str(count_transitions[50:53, 50:53]))

    # Make count_transitions row-stochastic
    row_sums = count_transitions.sum(axis=1)
    count_transitions = count_transitions / row_sums[:, np.newaxis]
    print('Sum of transitions should be one along rows: \n' + str(count_transitions.sum(axis=1)))

    note_hmm.transmat_ = count_transitions

    # Compute start probabilities as simply probability of presence of given note
    # start_probs = np.zeros(89)

    start_probs = np.bincount(np.ravel(target_corpus))
    start_probs = np.array(start_probs, dtype=np.float64)

    # Make start probabilities row-stochastic
    row_sum = start_probs.sum()
    start_probs = start_probs / row_sum
    # print('start_probs: ' + str(start_probs))

    note_hmm.startprob_ = start_probs

    # Compute Gaussians for each note
    means = np.zeros((NUM_NOTES, NUM_NOTES))
    covs = np.zeros((NUM_NOTES, NUM_NOTES, NUM_NOTES))

    for note_i in range(NUM_NOTES):

        probs = X_train[np.where(Y_train == note_i)]

        mean = np.mean(probs, axis=0)
        cov = np.cov(probs.T)

        # print('Indices of note ' + str(note_i) + ': ' + str(np.argwhere(Y_train == note_i)))
        # print('probs 1 to 3 for note ' + str(note_i) + ': ' + str(probs[0:3]))
        # print("mean: " + str(mean))

        means[note_i] = mean

        # Add identity here to make positive semidefinite
        covs[note_i] = 0.1 * np.identity(NUM_NOTES) + cov

    note_hmm.means_ = means
    note_hmm.covars_ = covs

    pickle.dump(note_hmm, open(output_model_file, 'wb'))


"""
Use trained hidden markov model to predict the sequence of notes which
generate the output probabilities in the TDNN. We perform this
decoding using the Viterbi algorithm.
"""
def hmm_predict(test_probs, target_notes, hmm_model, midi_file_name=None):

    X_test = pickle.load(open(test_probs, 'rb'))
    Y_test = pickle.load(open(target_notes, 'rb'))
    note_hmm = pickle.load(open(hmm_model, 'rb'))

    # Note: we expect a deprecation warning with the next command
    pred_hidden = note_hmm.predict(X_test)
    print('\n')

    print("Evaluating model on test set...")

    # Simple prediction by maximum of probabilities
    max_prob_pred = np.argmax(X_test, axis=1)
    compute_frame_accuracy(Y_test, max_prob_pred, 'tdnn')

    # Prediction using hmm decoding
    compute_frame_accuracy(Y_test, pred_hidden, 'hmm')

    # WER calculations
    compute_wer(Y_test, max_prob_pred, 'tdnn')
    compute_wer(Y_test, pred_hidden, 'hmm')

    if midi_file_name is not None:
        write_midi_mono(pred_hidden, midi_file_name)


"""
Compare two arrays of predicted notes, printing the percent accuracy
"""
def compute_frame_accuracy(actual, pred, model_name):
    matches = 0
    for i in range(len(pred)):
        if int(pred[i]) == int(actual[i]):
            matches += 1

    print('Accuracy on ' + model_name + ': ' + str(float(matches) / len(pred)))


"""
Calculates the word error rate (WER) for given lists of notes. Note: adapted 
from a tutorial on WER at https://martin-thoma.com/word-error-rate-calculation/
"""
def compute_wer(actual, pred, model_name):

    # Convert frame-by-frame listing of notes to an order listing of notes as words
    # (i.e. remove duplicate pitch frames when they occur consecutively)

    actual = actual.tolist()
    pred = pred.tolist()

    r = [note[0] for note in groupby(actual)]
    h = [note[0] for note in groupby(pred)]

    r = filter(lambda x: x != 88, r)
    h = filter(lambda x: x != 88, h)

    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    wer_score = float(d[len(r)][len(h)]) / len(r)
    print('WER on ' + model_name + ': ' + str(wer_score))
