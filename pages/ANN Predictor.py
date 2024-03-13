# Linear Regression on AirBNB cleaned data
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sma
from sklearn import metrics
import pickle
import sklearn.metrics as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import operator
import tensorflow as tf
import datetime
import os
from keras.callbacks import TensorBoard

@st.cache_data
def load_data():
    df = pd.read_csv('data/airbnbcleaned.csv')
    df = df.drop(['id','host_id'], axis=1)
    return df
df = load_data()

neighbourhood_mapping = {
    'Vesterbro-Kongens Enghave': 1,
    'Nørrebro': 2, 
    'Indre By': 3, 
    'Østerbro': 4,
    'Frederiksberg': 5,
    'Amager Vest': 6,
    'Amager st': 7,
    'Bispebjerg': 8,
    'Valby': 9, 
    'Vanløse': 10,
    'Brønshøj-Husum': 11
}


# Main
st.header('AirBNB Price Prediction with ANN')
st.subheader('Figure out the price of your AirBnB')
st.subheader('Artificial Neural Network predictor')
from keras.models import load_model

model = load_model('models/ann_model.h5')
# Create inputs for the user to enter values
availability_365 = st.number_input('Enter how many days your property is available out of 365')
accommodates = st.number_input('Enter how many people does the property fit')
neighbourhood_cleansed = st.number_input('Enter neighborhood by number as seen below')
st.markdown('1: Vesterbro-Kongens Enghave  | 2: Nørrebro  | 3: Indre By  |  4: Østerbro  |  5: Frederiksberg  |  6: Amager Vest  |  7: Amager  |  8: Bispebjerg  |  9: Valby  |  10: Vanløse  |  11: Brønshøj-Husum')
beds = st.number_input('how many beds/bedrooms?')
bathrooms = st.number_input('how many bathrooms?')
#neighbourhood_cleansed = neighbourhood_mapping.get(neighbourhood_name.lower())

# Button to make prediction
if st.button('Predict Price'):
    # Create a numpy array of the inputs
    input_data = np.array([[availability_365, accommodates, neighbourhood_cleansed, beds, bathrooms]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'Predicted Price: {round(prediction[0][0])},- for a minimum of a 3-5 day stay')
 


