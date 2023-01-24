'''
Author = Shreyas Pawar
Description = Script for creating a dataset from the downloaded json files
'''

import os
import pandas as pd
import json

'''
    Just make sure all the json files are in the directory
'''

# Get all the files in the directory
all_files = os.listdir()

# Filtering to only take the json files
file_list = [file for file in all_files if file.endswith(".json")]
# print(file_list)

# This list will hold all the dataframes for individual files
file_data_list = []

for file in file_list:
    with open(file,'r') as f:              
        # Parsing data into json format               
        data = json.loads(f.read())
    # Filtering json data to only get the dataframe with cols as indices of "aircraft" json values
    file_data_list.append(pd.json_normalize(data, record_path =['aircraft']))

# Creation of final dataset from individual dataframes
final_dataset = pd.concat(file_data_list)

final_dataset.to_csv("final_dataset.csv")

'''
For the final dataset manipulation, we have to take care of :

    mlat, tisb, nav_modes -> has values in lists 
    
    Most of the final cols dont have values for most rows so would have to drop them

'''