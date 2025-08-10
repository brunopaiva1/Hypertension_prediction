import pandas as pd
from sklearn.model_selection import train_test_split

db = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/hypertension_dataset.csv')

print(db.head())

y = db['hypertension']
df = db.drop(['hypertension'], axis='columns')
x = df

classes = db['hypertension'].value_counts()
print(classes)
