#!/Users/rudimiv/miniconda3/bin/python

import sys
import argparse

import h5py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D


class GoLPredictor:
    def __init__(self, width, height, filename, model_mode=False):
        self.width = width
        self.height = height

        if model_mode:
            self._load_model(filename)
        else:
            data = self._load_train_dataset(filename)
            self._train(30, 20, *data)


    def _load_train_dataset(self, filename):
        x, y = GoLPredictor._load_dataset(filename)
        train_size = 0.8

        X_train = np.pad(x[:int(train_size * x.shape[0]), :, :, np.newaxis], ((0,0),(1,1),(1,1),(0,0)), 'wrap')
        Y_train = y[:int(train_size * y.shape[0]), :, :, np.newaxis]

        X_val = np.pad(x[int(train_size * x.shape[0]):, :, :, np.newaxis], ((0,0),(1,1),(1,1),(0,0)), 'wrap')
        Y_val  = y[int(train_size * y.shape[0]):, :, :, np.newaxis]

        print(X_train.shape)

        return X_train, Y_train, X_val, Y_val

    def _load_test_dataset(self, filename):
        x, y =  GoLPredictor._load_dataset(filename)

        X_test = np.pad(x[:int(train_size * x.shape[0]), :, :, np.newaxis], ((0,0),(1,1),(1,1),(0,0)), 'wrap')
        Y_test = y[:int(train_size * y.shape[0]), :, :, np.newaxis]

        return X_test, Y_test
        

    @staticmethod
    def _load_dataset(filename):
        try:
            with h5py.File(filename, 'r') as f:
                x = np.array(f['x_train'])
                y = np.array(f['y_train'])
        except IOError:
            print('dataset file wasn\'t find')
            return None, None

        return x, y

    def _train(self, filters, hidden_dims, X_train, Y_train, X_val, Y_val, batch_size=50, epochs=2):
        # define model
        kernel_size = (3, 3)
        
        self._model = Sequential()
        self._model.add(Conv2D(
            filters, 
            kernel_size,
            padding='valid', 
            activation='relu', 
            input_shape=(self.width + 2, self.height + 2, 1)
        ))

        # self._model.add(Dense(hidden_dims, input_shape=(20,20, 1)))
        self._model.add(Dense(hidden_dims))
        self._model.add(Dense(1))
        self._model.add(Activation('sigmoid'))

        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self._model.summary())

        # train model
        self._model.fit(
            X_train, 
            Y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=(X_val, Y_val)
        )

        # return res
    
    def save(self, filename):
    

    def test(self, filename):
        data = _load_test_dataset(filename)

        res = self._model.evaluate(*data)

        print(res)
        print('Accuracy:', res[1])



def parse_cmd():
    def uint(value):
        ivalue = int(value)

        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    cmd_parser = argparse.ArgumentParser(description='Game Of Life next step predictor')
    cmd_parser.add_argument('-d', '--dataset', dest='dataset', type=str)
    cmd_parser.add_argument('-s', '--savemodel', dest='save', type=str)
    cmd_parser.add_argument('-t', '--test', dest='test', type=str)
    cmd_parser.add_argument('-m', '--model', dest='model', type=str)
    cmd_parser.add_argument('width', type=uint)
    cmd_parser.add_argument('height', type=uint)

    args = cmd_parser.parse_args(sys.argv[1:])

    print(args.width)
    print(args.height)

    return args.width, args.height, args.dataset, args.save, args.test, args.model

def main():
    width, height, f_dataset, model_save, f_test, model_load = parse_cmd()

    if model_load:
        golp = GoLPredictor(width, height, f_model, True)
    else:
        golp = GoLPredictor(width, height, f_dataset)

        if model_save:
            golp.save(model_save)

    if f_test:
        golp.test(f_test)

if __name__ == '__main__':
    main()
