'''
Author = Shreyas Pawar
Description = Script for handling the category, squawk and emergency column
'''

import pandas as pd
import numpy as np

# Pass the dataset to each fuction, and it should return a dataset

def handle_col_squawk(dataset):
  '''
  Do we really have to do this? We can just use the octal values?
  '''
  dataset.dropna(subset=['squawk'], inplace=True)  # Have to drop NA values before converting
  dataset['squawk'] = dataset['squawk'].apply(lambda x: int(str(int(x)),8))
  print(dataset["squawk"].describe())
  return dataset


def handle_col_category(dataset):
  '''
  Final changes 
  T1 = (A0+B0+C0) No ADS-B category information
  T2 = (A1+A2+A3) Weight upto 300000 lbs
  T3 = (A4)  High vortex large
  T4 = (A5) Heavy (> 300000 lbs)
  T5 = (A6) High performance (> 5g acceleration and 400 kts)
  T6 = (A7) Rotorcraft
  T7 = (B1+B2+B3+B4)
  T8 = (C1+C2+C3)

  '''
  # print(dataset["category"].count())
  # print(dataset['category'].value_counts())
  dataset = dataset[dataset["category"].isin(['A2','A5','A1','A3','A0','C0','C2','C3','A7','B2','C1','A4','A6','B4','B0','B1'])]
  dataset['category'] = dataset['category'].replace(['A0','B0','C0'], 'T1')
  dataset['category'] = dataset['category'].replace(['A1','A2','A3'], 'T2')
  dataset['category'] = dataset['category'].replace('A4', 'T3')
  dataset['category'] = dataset['category'].replace('A5', 'T4')
  dataset['category'] = dataset['category'].replace('A6', 'T5')
  dataset['category'] = dataset['category'].replace('A7', 'T6')
  dataset['category'] = dataset['category'].replace(['B1','B2','B3','B4'], 'T7')
  dataset['category'] = dataset['category'].replace(['C1','C2','C3'], 'T8')
  return dataset


# Should drop this column as most values have none in it and a lot are missing values
def handle_col_emergency(dataset):
  '''
  Check for value_counts before converting
  '''
  print(dataset["emergency"].unique())
  print(dataset['emergency'].value_counts())
  # dataset = pd.get_dummies(dataset, columns = ["emergency"])  # One hot encoding
  return dataset
