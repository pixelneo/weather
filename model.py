#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def windowize_dataset(dataset, learn_window, predict_window):
    dataset = dataset.shuffle(int(5*10e5))
    dataset = dataset.window(learn_window+predict_window, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(learn_window+predict_window))
    dataset = dataset.map(lambda w: (w[:learn_window], w[learn_window:]))
    dataset = dataset.shuffle(int(10^5))
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(4096)
    dataset = dataset.cache()
    return dataset

def load_data(file, learn_window=48, predict_window=24, train=0.7, dev=0.1, test=0.2):
    assert train + dev + test >= 0.999 and train + dev + test <= 1.001
    data = np.loadtxt(file, skiprows=1, delimiter=',', dtype=np.float32)
    data_length = len(data)
    dim = len(data[0])

    dataset = tf.data.Dataset.from_tensor_slices(data)

    trainset = dataset.take(int(train*data_length))
    testset = dataset.skip(int(train*data_length))
    devset = testset.take(int(dev*data_length))
    testset = testset.take(int(test*data_length))

    trainset = windowize_dataset(trainset, learn_window, predict_window)
    devset = windowize_dataset(devset, learn_window, predict_window)
    testset = windowize_dataset(testset, learn_window, predict_window)

    return trainset, devset, testset, dim

def feature_loss(feature=2,length=9):
    def mse_feature_loss(predicted_y, gold_y):
        # select = tf.one_hot(feature, length)
        # return tf.losses.MeanSquaredError(tf.multiply(select, predicted_y), tf.multiply(select, gold_y))
        return tf.losses.mean_squared_error(predicted_y[:,:,2], gold_y[:,:,2])
    return mse_feature_loss

def create_model(predict_window=24, predict_feature=2, init_lr=0.001, end_lr=0.00005, steps=10, dim=9):
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=[None, 9], return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.RepeatVector(predict_window),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(9))
    ])
    rate = 1 - end_lr/init_lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, steps, rate, staircase=True) 
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule), loss=feature_loss(predict_feature, dim))
    return model

def create_callbacks():
    chk = tf.keras.callbacks.ModelCheckpoint(filepath='./models/chk.ckpt', save_weights_only=True, verbose=1, save_freq='epoch')
    # tb = tf.keras.callbacks.TensorBoard(log_dir='./logs', write_graph=False, update_freq='epoch')

    return [chk]

if __name__ == '__main__':
    # train, dev, test, dim = load_data('data_frydlant.csv')
    train, dev, test, dim = load_data('data_frydlant.csv')

    model = create_model(dim=dim)
    callbacks = create_callbacks()
    model.fit(train, epochs=2, validation_data=dev, workers=8, use_multiprocessing=True, callbacks=callbacks)

    test2 = test.take(1)
    preds = model.predict(test2)
    real = []

    for x,y in test2:
        real.extend(y[:,:,2])
        print('x')
    preds2 = [] 
    for p in preds:
        preds2.append(p[:,2])

    for a,b in zip(real, preds2):
        print('pred: {} \nreal: {}\n---------------------------'.format('  '.join(map(lambda x: '{:.2f}'.format(x),b)), '  '.join(map(lambda x: '{:.2f}'.format(float(x)),list(a)))))
