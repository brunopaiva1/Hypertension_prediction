import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

try:
    db = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/hypertension_dataset.csv')
except FileNotFoundError:
    print("Arquivo não encontrado. Por favor, verifique o caminho para o arquivo hypertension_dataset.csv")
    data = {'Age': [45, 60, 35, 70, 50, 40, 80, 55, 65, 30],
            'Has_Hypertension': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]}
    db = pd.DataFrame(data)


classes = db['Has_Hypertension'].value_counts()
print("Distribuição de classes no dataset original:")
print(classes)


plt.figure(figsize=(8, 6))
sns.countplot(x='Has_Hypertension', data=db)
plt.title('Distribuição de Classes - Dataset Original')
plt.xlabel('Tem Hipertensão (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Não (0)', 'Sim (1)'])
plt.savefig('distribuicao_classes_original.png')
plt.close()


y = db['Has_Hypertension']
x = db.drop(['Has_Hypertension'], axis=1)

input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

output_train_df = pd.DataFrame(output_train)
plt.figure(figsize=(8, 6))
sns.countplot(x='Has_Hypertension', data=output_train_df)
# plt.title('Distribuição de Classes - Conjunto de Treino')
plt.xlabel('Tem Hipertensão (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Não (0)', 'Sim (1)'])
plt.savefig('distribuicao_classes_treino.png')
plt.close()

print("\nGráficos da distribuição de classes salvos como 'distribuicao_classes_original.png' e 'distribuicao_classes_treino.png'")