'''
Author = Shreyas Pawar
Description = Script for Logistic Regression Modelling
'''


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE

# Setting random seeds for everything
def set_random_seeds():
  random_seed = 1      # Using this seed throughout every part 
  np.random.seed(random_seed)
  return random_seed

# Shortening the dataset a little by balancing the "LIGHT" values by reducing to 800k values
def final_dataset(dataset):
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

# Scaling and normalizing the feartures
def create_sets_model(X, y, random_seed):
  '''
  Although the categories can be argued as not being ordinal, 
  using label encoder instead of One hot encoding as Regression works only with a single dimension target varaible
  '''

  label_encoder = LabelEncoder() 
  y = label_encoder.fit_transform(y)

  # Normalizing values
  scaler = MinMaxScaler(feature_range=(-1,1))     # Creating a -1,1 Normalizer 
  X = scaler.fit_transform(X)          # Normalizing all features and target variable in the dataset

  # Train_test split (70/30)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
  return X_train, X_test, y_train, y_test

# Initial model to test different optimizers
def model_simple_logistic(X_train, y_train, random_seed):
  model = LogisticRegression(max_iter=10000,multi_class='multinomial', solver='saga',random_state=random_seed)       # Creating the model with the tolerance
  model.fit(X_train, y_train)
  return model

# Getting the best features from the model performance
def best_model_features(X_train, y_train, X_test, y_test, random_seed):
  model = LogisticRegression(max_iter=10000,multi_class='multinomial', solver='newton-cg',random_state=random_seed)
  rfe_selector = RFE(estimator=model, n_features_to_select=4 , step = 1)
  rfe_selector.fit(X_train, y_train)
  newxtrain = rfe_selector.transform(X_train)
  newxtest = rfe_selector.transform(X_test)

  model.fit(newxtrain, y_train)

  y_pred = model.predict(newxtest)

  print("Number of features: 4")
  print("Test set MSE={1:e}"
  .format(metrics.mean_squared_error(y_test, y_pred)))
  print("\nUsing: {0}".format(X.columns[rfe_selector.get_support()]))
  return X.columns[rfe_selector.get_support()]

# Final model to run on whole dataset
def best_model(X_train, y_train, X_test, y_test, random_seed):
  model = LogisticRegression(max_iter=10000,multi_class='multinomial', solver='newton-cg',random_state=random_seed)
  model.fit(X_train, y_train)
  return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
  print("Number of iterations=", model.n_iter_)
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  print("MSE Train = %f, MSE Test = %f, Classification Accuracy = %f" %
  (metrics.mean_squared_error(y_train, y_pred_train), metrics.mean_squared_error(y_test, y_pred_test), metrics.accuracy_score(y_test, y_pred_test)))
  confusion_matrix = metrics.plot_confusion_matrix(model, X_test, y_test, cmap="Blues")
  confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
  print(confusion_matrix)

# Used in 2 iterations to calculate the best number of features to use
def feature_selection_model(X_train, y_train, X_test, y_test, random_seed):
  # n_features = [5,10,15,20]
  n_features = [3,4,6,7,8]
  model = LogisticRegression(max_iter=10000,multi_class='multinomial', solver='newton-cg',random_state=random_seed)

  for feature in n_features:

    rfe_selector = RFE(estimator=model, n_features_to_select = feature, step = 1)
    rfe_selector.fit(X_train, y_train)
    newxtrain = rfe_selector.transform(X_train)
    newxtest = rfe_selector.transform(X_test)

    model.fit(newxtrain, y_train)

    y_pred_train = model.predict(newxtrain)
    y_pred_test = model.predict(newxtest)

    print("Number of features:", feature)
    print("Train set MSE={1:e}, Test set MSE={1:e}"
    .format(metrics.mean_squared_error(y_train, y_pred_train), 
            metrics.mean_squared_error(y_test, y_pred_test)))

random_seed = set_random_seeds()

dataset = pd.read_csv("raw_data_20c_3y_clean.csv")
dataset = final_dataset(dataset)

X = dataset.drop(["category"], axis=1)
y = dataset["category"]

# best_features = best_model_features(X_train, y_train, random_seed)

# The below values are obtained from the previous iteration of above function 
best_features = ['alt_geom', 'gs', 'nac_v', 'messages']

X = X[best_features]
X_train, X_test, y_train, y_test = create_sets_model(X, y, random_seed)

final_model = best_model(X_train, y_train, X_test, y_test, random_seed)
evaluate_model(final_model, X_train, X_test, y_train, y_test)