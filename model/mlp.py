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

output_train_numeric = output_train.copy()
output_train_numeric['Has_Hypertension'] = output_train['Has_Hypertension'].map({'Yes': 1, 'No': 0})

output_test_numeric = output_test.copy()
output_test_numeric['Has_Hypertension'] = output_test['Has_Hypertension'].map({'Yes': 1, 'No': 0})

input_test_numeric = pd.get_dummies(input_test, drop_first=True)
train_cols = input_train.columns
input_test_aligned = input_test_numeric.reindex(columns=train_cols, fill_value=0)

scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test_aligned)

pd.DataFrame(input_train_scaled).to_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_scaled.csv', index=False)
pd.DataFrame(input_test_scaled).to_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test_scaled.csv', index=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(input_train_scaled, output_train_numeric, epochs=100, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('/home/bruno/Hypertension_prediction/loss_plot.png')
plt.show()  

pd.DataFrame(history.history).to_csv('/home/bruno/Hypertension_prediction/history.csv', index=False)

model.save('/home/bruno/Hypertension_prediction/model/hypertension_model.h5')