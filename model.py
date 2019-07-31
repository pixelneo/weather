#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def windowize_dataset(dataset, learn_window, predict_window):
    dataset = dataset.shuffle(int(5*10e5))
    dataset = dataset.window(learn_window+predict_window, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(learn_window+predict_window))
    dataset = dataset.map(lambda w: (tf.expand_dims(w[:learn_window], -1), w[learn_window:]))
    dataset = dataset.shuffle(int(10^5))
    dataset = dataset.batch(16)
    dataset = dataset.cache()
    return dataset

def load_data(file, learn_window=48, predict_window=24, train=0.7, dev=0.1, test=0.2):
    assert train + dev + test >= 0.999 and train + dev + test <= 1.001
    data = np.loadtxt(file, dtype=np.float32)
    data_length = len(data)
    data = tf.constant(data, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    trainset = dataset.take(int(train*data_length))
    testset = dataset.skip(int(train*data_length))
    devset = testset.take(int(dev*data_length))
    testset = testset.take(int(test*data_length))

    trainset = windowize_dataset(trainset, learn_window, predict_window)
    devset = windowize_dataset(devset, learn_window, predict_window)
    testset = windowize_dataset(testset, learn_window, predict_window)

    return trainset, devset, testset

def create_model(predict_window=24):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(predict_window)
    ])
    model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
    return model

if __name__ == '__main__':
    train, dev, test = load_data('frydlant_teplota2.txt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/chk.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)
    
    model = create_model()
    model.fit(train.take(5), epochs=1, validation_data=dev,workers=8, use_multiprocessing=True, callbacks=[cp_callback])

    test2 = dev.take(1)
    preds = model.predict(test2)
    print(preds)
    a, b = test2
    exit()
    for val, p in zip(np.array(test2), preds):
        _, gold = val
        print(gold)
        print(p)
        print('----')
