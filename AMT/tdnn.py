import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import load_model

from write_midi_mono import write_midi_mono

NUM_CLASSES = 89 # 88 notes plus one class for silence
NUM_MFCC = 30
NUM_CEPS = 13


"""
Train a time-delay neural network (implemented with Keras convolutional layers)
to predict the probability of a given note's presence at particular time frames
based on surrounding MFCCs
"""
def tdnn_train(tdnn_feat_train, tdnn_target_train, model_name):

    X_train = pickle.load(open(tdnn_feat_train, 'rb'))
    Y_train = pickle.load(open(tdnn_target_train, 'rb'))

    num_samples = X_train.shape[0]
    random_indexes = np.random.permutation(num_samples)

    X_train = X_train[random_indexes, :]
    Y_train = Y_train[random_indexes]

    print("Training data shape: " + str(X_train.shape))
    print("Training target shape: " + str(Y_train.shape))

    model = tdnn_model()

    # Train the model with stochastic gradient descent using the following learning rate and schedule
    lr = 0.001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    def lr_schedule(epoch):
        return lr * (0.1 ** int(epoch / 10))

    batch_size = 32
    epochs = 5

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('models/model.h5', save_best_only=True)]
              )

    model.save(model_name)


"""
Use trained TDNN to predict probabilities of each note's presence for each time 
frame given the surrounding MFCCs
"""
def tdnn_predict(model_name, tdnn_feat_train, tdnn_target_train, tdnn_feat_test, tdnn_target_test, midi_file_name=None):

    X_train = pickle.load(open(tdnn_feat_train, 'rb'))
    Y_train = pickle.load(open(tdnn_target_train, 'rb'))

    X_test = pickle.load(open(tdnn_feat_test, 'rb'))
    Y_test = pickle.load(open(tdnn_target_test, 'rb'))

    print("Testing data shape: " + str(X_test.shape))
    print("Testing target shape: " + str(Y_test.shape))

    model = load_model(model_name)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # Notes predicted by simple max of probabilities
    # train_pred_notes = np.argmax(Y_train_pred, axis=1)
    test_pred_notes = np.argmax(Y_test_pred, axis=1)
    train_act_notes = np.argmax(Y_train, axis=1)
    test_act_notes = np.argmax(Y_test, axis=1)

    if midi_file_name is not None:
        write_midi_mono(test_pred_notes, midi_file_name)

    pickle.dump(Y_train_pred, open("data/hmm/train_probs.pkl", 'wb'))
    pickle.dump(train_act_notes, open("data/hmm/train_notes.pkl", 'wb'))

    pickle.dump(Y_test_pred, open("data/hmm/test_probs.pkl", 'wb'))
    pickle.dump(test_act_notes, open("data/hmm/test_notes.pkl", 'wb'))


"""
Returns an untrained time-delay neural network model in Keras using 
one-dimensional convolutional layers across the time axis. Note 
this model is adapted from an image traffic sign recognition convolutional 
neural net at https://chsasank.github.io/keras-tutorial.html
"""
def tdnn_model():

    model = Sequential()

    model.add(Conv1D(64, (2), padding='same',
                     input_shape=(NUM_CEPS, NUM_MFCC),
                     activation='relu'))
    model.add(Conv1D(64, (2), activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, (3), padding='same',
                     activation='relu'))
    model.add(Conv1D(64, (3), activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, (4), padding='same',
                     activation='relu'))
#     model.add(Conv1D(128, (2), activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model