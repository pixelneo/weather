#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def windowize_dataset(dataset, learn_window, predict_window):
    dataset = dataset.window(learn_window+predict_window, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(learn_window+predict_window))
    dataset = dataset.map(lambda w: (w[:learn_window], w[learn_window:]))
    dataset = dataset.shuffle(int(10^5))
    dataset = dataset.batch(128, drop_remainder=True)
    dataset = dataset.shuffle(int(10^5))
    dataset = dataset.prefetch(4096)
    dataset = dataset.cache()
    return dataset

def load_data(file, learn_window=72, predict_window=24, train=0.9, dev=0.05, test=0.05):
    assert train + dev + test >= 0.999 and train + dev + test <= 1.001
    data = np.loadtxt(file, skiprows=1, delimiter=',', dtype=np.float32)
    data = tf.constant(data)
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

def windowize_dataset_alt(dataset, learn_window, predict_window):
    x, y = [], []
    for i in range(len(dataset)-learn_window-predict_window-1):
        x.append(dataset[i:i+learn_window])
        y.append(dataset[i+learn_window:i+learn_window+predict_window])
    p = np.random.permutation(len(x))
    x = np.array(x)
    y = np.array(y)

    x = x[p]
    y = y[p]
    return x,y

def load_data_alt(file, learn_window=72, predict_window=24, train=0.9, dev=0.05, test=0.05):
    assert train + dev + test >= 0.999 and train + dev + test <= 1.001
    data = np.loadtxt(file, skiprows=1, delimiter=',', dtype=np.float32)
    data_length = len(data)
    dim = len(data[0])

    trainset = data[:int(train*data_length)]
    testset = data[int(train*data_length):int((train+test)*data_length)]
    devset = data[int((train+test)*data_length):]

    trainset = windowize_dataset_alt(trainset, learn_window, predict_window)
    testset = windowize_dataset_alt(testset, learn_window, predict_window)
    devset = windowize_dataset_alt(devset, learn_window, predict_window)

    return trainset, devset, testset, dim




def lr_scheduler(initial=1e-4, final=4e-6):
    def s(epoch):
        return max(final, initial/((epoch+1)*2))
    return s

def feature_loss(feature=2,length=9):
    def mse_feature_loss(predicted_y, gold_y):
        # select = tf.one_hot(feature, length)
        # return tf.losses.MeanSquaredError(tf.multiply(select, predicted_y), tf.multiply(select, gold_y))
        # weight = np.arange(24,12,-0.5)
        return tf.losses.mean_squared_error(predicted_y[:,:,2], gold_y[:,:,2]) #* weight
    return mse_feature_loss

def create_model(predict_window=24, predict_feature=2, init_lr=0.0001, end_lr=0.00001, steps=10, dim=9):
    
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, reset_after=True, recurrent_activation='sigmoid', activation='relu', input_shape=[None, 9], return_sequences=True),
        # tf.keras.layers.GRU(64, activation='relu', return_sequences=True),
        tf.keras.layers.GRU(64, reset_after=True, recurrent_activation='sigmoid', activation='relu'),
        tf.keras.layers.RepeatVector(predict_window),
        tf.keras.layers.GRU(64, reset_after=True, recurrent_activation='sigmoid', activation='relu', return_sequences=True),
        tf.keras.layers.GRU(64, reset_after=True, recurrent_activation='sigmoid', activation='relu', return_sequences=True),
        # tf.keras.layers.GRU(64, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(9))
    ])
    model.compile(optimizer=tf.optimizers.Adam(), loss=feature_loss(predict_feature, dim))
    return model

def teacher_forcing(predict_window=24, predict_feature=2, init_lr=0.0001, end_lr=0.00001, steps=10, dim=9):
    # tf.keras.backend.set_floatx('float16')
    encoder_in = tf.keras.layers.Input(batch_shape=(128, 72, 9))
    encoder = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, recurrent_activation='sigmoid', activation='relu', return_sequences=True))(encoder_in)
    encoder_out, f, b = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, recurrent_activation='sigmoid', return_state=True))(encoder)
    hidden = [f,b]

    # decoder
    decoder_in = tf.keras.layers.Input(batch_shape=(128, 72, 9))
    decoder = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, recurrent_activation='sigmoid', activation='relu', return_sequences=True))(decoder_in, initial_state=hidden)
    decoder_out, f2, b2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, recurrent_activation='sigmoid', return_state=True))(decoder)

    decoder_dense = tf.keras.layers.Dense(128, activation='relu')(decoder_out)
    decoder_dense = tf.keras.layers.Dense(predict_window, activation='softmax')(decoder_dense)

    model = tf.keras.models.Model([encoder_in, decoder_in],decoder_dense)



def create_callbacks():
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler())
    chk = tf.keras.callbacks.ModelCheckpoint(filepath='./models/chk.ckpt', save_weights_only=True, verbose=1, save_freq='epoch')
    # tb = tf.keras.callbacks.TensorBoard(log_dir='./logs', write_graph=False, update_freq='epoch')

    return [chk, lr_schedule]
def train(model, data):
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y), dim = data
    callbacks = create_callbacks()
    model.fit(train_x, train_y, epochs=12, batch_size=128, validation_data=(dev_x, dev_y), workers=8, use_multiprocessing=True, callbacks=callbacks)

if __name__ == '__main__':
    # train, dev, test, dim = load_data('data_frydlant.csv')
    # data = load_data_alt('data_frydlant.csv')
    # (train_x, train_y), (dev_x, dev_y), (test_x, test_y), dim = data
    # model = create_model(dim=dim)
    # model.load_weights('./models/chk.ckpt')
    teacher_forcing() 

    exit()
    train(model, data)
    

    preds = model.predict([test_x[:512]])

    real = []
    for x,y in zip(test_x[:512], test_y[:512]):
        real.append(y[:,2])
    preds2 = [] 
    for p in preds:
        preds2.append(p[:,2])

    for a,b in zip(real, preds2):
        print('pred: {} \nreal: {}\n---------------------------'.format('  '.join(map(lambda x: '{:.2f}'.format(x),b)), '  '.join(map(lambda x: '{:.2f}'.format(float(x)),list(a)))))
