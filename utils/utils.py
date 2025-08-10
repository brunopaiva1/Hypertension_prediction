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

input_train = pd.DataFrame(input_train)
input_train.to_csv('/home/bruno/Hypertension_prediction/dataset/train/input_train.csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('/home/bruno/Hypertension_prediction/dataset/test/input_test.csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('/home/bruno/Hypertension_prediction/dataset/train/output_train.csv', index=False)

output_test = pd.DataFrame(output_test)
output_test.to_csv('/home/bruno/Hypertension_prediction/dataset/test/output_test.csv', index=False)
print("Data has been split and saved to CSV files.")