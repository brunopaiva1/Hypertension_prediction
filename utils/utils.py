import pandas as pd
from sklearn.model_selection import train_test_split

db = pd.read_csv('/home/bruno/Hypertension_prediction/dataset/hypertension_dataset.csv')

print(db.head())

y = db['Has_Hypertension']
df = db.drop(['Has_Hypertension'], axis='columns')
x = df

classes = db['Has_Hypertension'].value_counts()
print(classes)

input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(input_train)}")
print(f"Test set size: {len(input_test)}")
