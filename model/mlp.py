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