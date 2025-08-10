from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf

model = tf.keras.models.load_model('/home/bruno/Hypertension_prediction/model/hypertension_model.h5')

saida_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/output_test.csv')
entrada_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test_scaled.csv')

saida_test_numeric = saida_test['Has_Hypertension'].map({'Yes': 1, 'No': 0})

history = pd.read_csv('/home/bruno/Hypertension_prediction/history.csv')

plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/bruno/Hypertension_prediction/evaluation_loss_plot.png')
plt.close() 

output_model = model.predict(entrada_test)

output_model_classes = (output_model > 0.5).astype("int32")

print("Classification Report:")
print(classification_report(saida_test_numeric, output_model_classes))

print('Accuracy:', accuracy_score(saida_test_numeric, output_model_classes))
print('Precision:', precision_score(saida_test_numeric, output_model_classes))
print('Recall:', recall_score(saida_test_numeric, output_model_classes))
print('F1 Score:', f1_score(saida_test_numeric, output_model_classes))

cm = confusion_matrix(saida_test_numeric, output_model_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Hypertension', 'Hypertension'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('/home/bruno/Hypertension_prediction/confusion_matrix.png') 
plt.close()