import os
import tensorflow as tf
import pennylane as qml
from pennylane.qnn.keras import KerasLayer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.regularizers import l2

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

print("\n--- Iniciando Benchmark de Modelos Clássicos ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42, algorithm='SAMME'),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n--- Treinando e Avaliando: {name} ---")
    model.fit(input_train, output_train_numeric.values.ravel())
    predictions = model.predict(input_test_aligned)
    accuracy = accuracy_score(output_test_numeric, predictions)
    f1 = f1_score(output_test_numeric, predictions)
    results.append({
        "Modelo": name,
        "Acurácia": accuracy,
        "F1-Score": f1
    })
    print(classification_report(output_test_numeric, predictions, target_names=['No Hypertension', 'Hypertension']))

print("\n\n--- Resumo do Benchmark de Modelos Clássicos ---")
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print(results_df)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
input_dim = input_train_scaled.shape[1]
n_qubits = 4

print("\n--- Treinando Modelo Clássico (MLP) ---")
model_classic = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
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
print(f"classic_mlp - Test Loss: {loss_c:.4f}, Test Accuracy: {acc_c:.4f}")

print("\n--- Definindo e Desenhando Circuito Quântico ---")

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

fig, ax = qml.draw_mpl(quantum_circuit, style='pennylane')(tf.zeros(n_qubits), tf.zeros((1, n_qubits, 3)))
plt.title(f'Circuito Quântico Híbrido com {n_qubits} Qubits', fontsize=14)
plt.tight_layout()
circuit_image_path = '/home/bruno/Hypertension_prediction/plot-teste/circuito_quantico_4qubits.png'
plt.savefig(circuit_image_path)
print(f"Imagem do circuito salva em: {circuit_image_path}")
plt.close()

print("\n--- Treinando Modelo Híbrido ---")

weight_shapes = {"weights": (1, n_qubits, 3)}
quantum_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

model_hybrid = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'), #, kernel_regularizer=l2(0.01)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_qubits, activation='tanh'),
    quantum_layer,
    # tf.keras.layers.Reshape((1,)),
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
print(f"hybrid_qml - Test Loss: {loss_h:.4f}, Test Accuracy: {acc_h:.4f}")

print("\n--- Avaliação Detalhada do Modelo Híbrido ---")

predictions_hybrid = model_hybrid.predict(input_test_scaled)
predicted_classes_hybrid = (predictions_hybrid > 0.5).astype("int32")
report = classification_report(output_test_numeric.values, predicted_classes_hybrid, target_names=['No Hypertension', 'Hypertension'])
print("Classification Report:\n", report)

print("\n--- Gerando Gráficos ---")

# --- Matriz de Confusão do Modelo Clássico (MLP) ---
# NOVO: Faz as previsões com o modelo clássico
predictions_classic = model_classic.predict(input_test_scaled)
predicted_classes_classic = (predictions_classic > 0.5).astype("int32")

# NOVO: Calcula e plota a matriz de confusão
cm_classic = confusion_matrix(output_test_numeric.values, predicted_classes_classic)
disp_classic = ConfusionMatrixDisplay(confusion_matrix=cm_classic, display_labels=['No Hypertension', 'Hypertension'])
disp_classic.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - Modelo Clássico (MLP)')
plt.savefig('/home/bruno/Hypertension_prediction/plot-teste/confusion_matrix_classic.png')
plt.close()
print("Matriz de confusão do modelo clássico salva.")


# --- Matriz de Confusão do Modelo Híbrido ---
# (Esta parte você já tinha, apenas a mantive para ficar completo)
cm_hybrid = confusion_matrix(output_test_numeric.values, predicted_classes_hybrid)
disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=['No Hypertension', 'Hypertension'])
disp_hybrid.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - Modelo Híbrido')
plt.savefig('/home/bruno/Hypertension_prediction/plot-teste/confusion_matrix_hybrid.png')
plt.close()
print("Matriz de confusão do modelo híbrido salva.")


# --- Gráfico de Comparação de Perda (Loss) ---
# (Esta parte você já tinha também)
plt.figure(figsize=(10, 6))
plt.plot(history_classic.history['val_loss'], label='Classic MLP Val Loss')
plt.plot(history_hybrid.history['val_loss'], label='Hybrid QML Val Loss')
plt.title('Comparação da Perda de Validação (Loss)')
plt.ylabel('Perda (Loss)')
plt.xlabel('Época (Epoch)')
plt.legend()
plt.grid(True)
plt.savefig('/home/bruno/Hypertension_prediction/plot-teste/loss_comparison.png')
plt.close()
print("Gráfico de comparação de perda salvo.")


print("\nProcesso concluído. Saída e gráficos salvos com sucesso!")
