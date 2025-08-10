import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

input_train = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_balanced.csv')
output_train = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/output_train_balanced.csv')

input_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/output_test.csv')

scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test)

pd.DataFrame(input_train_scaled).to_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_scaled.csv', index=False)
pd.DataFrame(input_test_scaled).to_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test_scaled.csv', index=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(input_train_scaled, output_train, epochs=100, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping])