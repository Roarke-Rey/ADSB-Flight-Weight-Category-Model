# ADSB-Flight-Weight-Category-Model
An iterative ANN and Regression model to predict the flight weight category based on public ADSB records for all flights in the air.

## Dataset Preparation
The final dataset was based on the ADSB sample flight records available at https://www.adsbexchange.com/data-samples/.
Python scripts were developed to download a list of available data and then subsample the available files given a date range and count of files per day. These scripts were then used to download and merge over five million individual ADS-B records over a three-year period between January 2019 and December 2022

## Models
Two models of Regression and Artificial Neural Networks were iteratively developed and tuned by me, whose performance was then compared with several other models developed by teammates.

### ANN Model

The final architecture of the ANN was chosen based on Occum's razor principle where the number of nodes in a layer were increased rather than increasing the number of layers to obtain the best architecture.

<div align="center">Architectural performance comparison of ANN <br>

<img width="367" alt="image" src="https://user-images.githubusercontent.com/57321224/214414800-0e89e0b7-4cec-49c5-8c65-20df633994a4.png">
</div>
<br>
The overfitting problem of a 150x16x8 ANN was also stopped using the early stopping mechanism<br> <br>
<div align="center"> Training curve for the model <br>
<img width="288" alt="image" src="https://user-images.githubusercontent.com/57321224/214417556-be31f6cd-809c-4ee3-ad5e-00fcedd14bdf.png">
</div>

### Regression Model
This model was trained similarly to the ANN model on the three different datasets, with all model tuning done 
on the randomly sampled dataset, to get the best values for hyperparameters. 

<div align="center">
Model Performance with different optimizers <br>
<img width="299" alt="image" src="https://user-images.githubusercontent.com/57321224/214418045-3f0645db-4856-4140-b057-8074d7299558.png">
</div>
<br>
Also implemented feature selection techniques to narrow down the dataset features for higher performance
<br>
<div align="center">
Feature selection Performance<br>
<img width="299" alt="image" src="https://user-images.githubusercontent.com/57321224/214418261-f556e5f4-a5cf-49a2-a5d2-76184b1684d0.png">

</div>

## Model Performance Comparison

A derived metric from the confusion matrix was used to compare the performance of every model, where the metric was given by: 
<div align="center">
<img width="594" alt="image" src="https://user-images.githubusercontent.com/57321224/214419231-2945e86b-59c8-42f3-ad8e-0a484f63bc36.png">
</div>
The scores were then plotted against the models, to quantify the best performing model. 
<div align="center"> 
<img width="361" alt="image" src="https://user-images.githubusercontent.com/57321224/214419067-f3223708-dcdc-426f-a327-62c146bf547b.png">
</div>
