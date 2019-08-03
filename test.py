#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# create dataset
x = np.random.normal(size=[4,5])
y = np.random.normal(size=4)

dataset = tf.data.Dataset.from_tensor_slices((x,y))
dataset = dataset.batch(2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
model.fit(dataset, epochs=2)

