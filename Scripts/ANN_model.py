'''
Author = Shreyas Pawar
Description = Script for ANN modelling
'''


import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sn

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def random_sample_dataset(dataset):
  sampled_dataset = dataset.sample(n=100000,random_state=random_seed)
  # sampled_dataset.to_csv("randomly_sampled_dataset.csv")        # Storing to use again later
  return sampled_dataset

def eqi_sampled_dataset(dataset):
  category_columns = dataset["category"].unique()
  sampled_dataset = pd.DataFrame(columns=dataset.columns)
  for category in category_columns:
    dataset_slice = dataset[dataset["category"] == category]
    if dataset_slice.shape[0] < 12000:
      sampled_dataset = sampled_dataset.append(dataset_slice)
    else:
      sampled_dataset = sampled_dataset.append(dataset_slice.sample(n=12000, random_state=random_seed))
  # sampled_dataset.to_csv("equally_sampled_dataset.csv")         # Storing to use again later
  return sampled_dataset

def final_dataset(dataset):
  # Shortening the dataset a little by balancing the "LIGHT" values by reducing to 800k values
  dataset = dataset.drop(["hex","type","version"], axis=1)
  category_columns = dataset["category"].unique()
  sampled_dataset = pd.DataFrame(columns=dataset.columns)
  for category in category_columns:
    dataset_slice = dataset[dataset["category"] == category]
    if category == "LIGHT":
      sampled_dataset = sampled_dataset.append(dataset_slice.sample(n=800000, random_state=random_seed))
    else:
      sampled_dataset = sampled_dataset.append(dataset_slice)
  return sampled_dataset

def evaluate_model(model, X_train, X_test, y_train, y_test):
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  print("TF: Train MSE: {:7.5f}"
  .format(mean_squared_error(y_train, y_pred_train)))
  print("TF: Test MSE: {:7.5f}".format(mean_squared_error(y_test, y_pred_test)))
  final_confusion_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
  df_cm = pd.DataFrame(final_confusion_matrix)
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=True, cmap="Blues", robust=True, fmt="d")
  return final_confusion_matrix

random_seed = 1      # Using this seed throughout every part 
tf.random.set_seed(random_seed)    # Setting the seed for tensorflow models too
np.random.seed(random_seed)

dataset = pd.read_csv("raw_data_20c_3y_clean.csv")
dataset = final_dataset(dataset)
# dataset = pd.read_csv("equally_sampled_dataset.csv")
# dataset = pd.read_csv("randomly_sampled_dataset.csv")

# print(dataset.shape)

# print(dataset.describe())
# print("Checking for Null values:", dataset.isnull().values.any())

# print(dataset["category"].value_counts())

X = dataset.drop(["category"], axis=1)
y = dataset["category"]

one_hot_encoder = OneHotEncoder(sparse=False)    # Disabling the sparse flag so that it returns an array instead of Sparse Matrix
y = one_hot_encoder.fit_transform(np.array(y).reshape(-1,1))

# Normalizing values
scaler = MinMaxScaler(feature_range=(-1,1))     # Creating a -1,1 Normalizer 
X = scaler.fit_transform(X)          # Normalizing all features and target variable in the dataset

# Train_test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# print(X_train.shape)

# Modelling
epochs = 200
inputwidth = X_train.shape[1]

# Tried using various architectures here to check the model performance
model3 = Sequential()
model3.add(Dense(150, input_shape=(X.shape[1],), activation='tanh')) # input shape is (features,)
model3.add(Dense(16, activation='tanh'))  # Stuck to the 'tanh' activation functions for the input and hidden layers
model3.add(Dense(8, activation='softmax'))  # Experimented with output layer activation functions
model3.summary()

model3.compile(optimizer='adam',      # Firm believer of the adam optimizer
              loss="categorical_crossentropy", 
              metrics=['accuracy'])  


# Implementing early stopping on the validation loss so it stops if there are no good changes in 20 epochs, or the generalization gap increases
es = EarlyStopping(monitor='val_loss',                # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
                  mode='min',
                  patience=20, 
                  restore_best_weights=True) # important - otherwise you just return the last weigths..

history3 = model3.fit(X_train,
                    y_train,
                    epochs=epochs,
                    callbacks=[es],
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.2,     # Using a 80-20 validation split
                    verbose=1)

# Plotting learning history

def plot_loss(history):         # Plotting graphs - https://www.tensorflow.org/tutorials/keras/regression
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0])
  plt.xlabel('Epoch')
  plt.ylabel('Error MSE')
  plt.legend()
  plt.grid(True)

plot_loss(history3)

confusion_matrix = evaluate_model(model3, X_train, X_test, y_train, y_test)