import os
import tensorflow as tf
import pennylane as qml
from pennylane.qnn import KerasLayer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

input_train = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_balanced.csv')
output_train = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/output_train_balanced.csv')
input_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/output_test.csv')

output_train_numeric = output_train['Has_Hypertension'].map({'Yes': 1, 'No': 0})
output_test_numeric = output_test['Has_Hypertension'].map({'Yes': 1, 'No': 0})

input_test_numeric = pd.get_dummies(input_test, drop_first=True)
train_cols = input_train.columns
input_test_aligned = input_test_numeric.reindex(columns=train_cols, fill_value=0)

scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test_aligned)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
input_dim = input_train_scaled.shape[1]
n_qubits = 4

print("\n--- Treinando Modelo Clássico ---")
model_classic = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

opt_classic = tf.keras.optimizers.Adam(learning_rate=0.001)
model_classic.compile(optimizer=opt_classic, loss='binary_crossentropy', metrics=['accuracy'])
history_classic = model_classic.fit(
    input_train_scaled, output_train_numeric,
    epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping], verbose=1
)
loss_c, acc_c = model_classic.evaluate(input_test_scaled, output_test_numeric, verbose=0)
print(f"classic - Test Loss: {loss_c:.4f}, Test Accuracy: {acc_c:.4f}")

print("\n--- Treinando Modelo Híbrido ---")

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (1, n_qubits, 3)}
quantum_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=1)

model_hybrid = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(n_qubits, activation='tanh'),
    quantum_layer,
    tf.keras.layers.Reshape((1,)),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

opt_hybrid = tf.keras.optimizers.Adam(learning_rate=0.001)
model_hybrid.compile(optimizer=opt_hybrid, loss='binary_crossentropy', metrics=['accuracy'])
history_hybrid = model_hybrid.fit(
    input_train_scaled, output_train_numeric,
    epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping], verbose=1
)
loss_h, acc_h = model_hybrid.evaluate(input_test_scaled, output_test_numeric, verbose=0)
print(f"hybrid - Test Loss: {loss_h:.4f}, Test Accuracy: {acc_h:.4f}")

print("\n--- Avaliação Detalhada do Modelo Híbrido ---")

predictions_hybrid = model_hybrid.predict(input_test_scaled)
predicted_classes_hybrid = (predictions_hybrid > 0.5).astype("int32")
report = classification_report(output_test_numeric.values, predicted_classes_hybrid, target_names=['No Hypertension', 'Hypertension'])
print("Classification Report:\n", report)

print("\n--- Gerando Gráficos ---")

cm = confusion_matrix(output_test_numeric.values, predicted_classes_hybrid)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Hypertension', 'Hypertension'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - Modelo Híbrido')
plt.savefig('/home/bruno/Hypertension_prediction/confusion_matrix_hybrid.png')
plt.close() 

plt.figure(figsize=(12, 8)) 

plt.plot(history_classic.history['val_loss'], label='Classic Val Loss', color='blue')
plt.plot(history_classic.history['loss'], label='Classic Train Loss', color='cyan', linestyle='--') 

plt.plot(history_hybrid.history['val_loss'], label='Hybrid Val Loss', color='orange')
plt.plot(history_hybrid.history['loss'], label='Hybrid Train Loss', color='red', linestyle='--') 

plt.title('Comparação Completa de Perda (Loss)')
plt.ylabel('Perda (Loss)')
plt.xlabel('Época (Epoch)')
plt.legend()
plt.grid(True)
plt.savefig('/home/bruno/Hypertension_prediction/loss_comparison_full.png')
plt.close()

print("\nProcesso concluído. Gráficos salvos com sucesso!")