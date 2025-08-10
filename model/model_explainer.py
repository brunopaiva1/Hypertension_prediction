import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

model = tf.keras.models.load_model('/home/bruno/Hypertension_prediction/model/hypertension_model.h5')

# selected_features = ['age',  'Salt_intake', 'Stress_Score', 'BP_History', 'Sleep_Duration', 'BMI', 'Medication', 'Family_History', 
#                      'Exercise_Level', 'Smoking_Status']

input_test = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test_scaled.csv')
input_train = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_scaled.csv')

feature_names_df = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train_balanced.csv')
correct_feature_names = feature_names_df.columns.tolist()

input_test.columns = correct_feature_names
input_train.columns = correct_feature_names

background_data = shap.sample(input_test, 100)
explainer = shap.KernelExplainer(model.predict, background_data)

shap_values = explainer.shap_values(input_test.iloc[:10])

print("SHAP values calculated.")

shap_values = np.array(shap_values).squeeze(-1)

shap.summary_plot(shap_values, input_test.iloc[:10], plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('/home/bruno/Hypertension_prediction/model/shap_summary_plot.png')
plt.close()

shap.dependence_plot('BP_History_Normal', shap_values, input_test.iloc[:10], show=False)
plt.tight_layout()
plt.savefig('/home/bruno/Hypertension_prediction/model/shap_dependence_plot_BP_History_Normal.png')
plt.close()

print("Gráficos do SHAP salvos com sucesso.")

