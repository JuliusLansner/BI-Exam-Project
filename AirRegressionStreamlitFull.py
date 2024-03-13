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

@st.cache
def load_data():
    df = pd.read_csv('data/airbnbcleaned.csv')
    df = df.drop(['id','host_id'], axis=1)
    return df
df = load_data()

# Main
st.header('AirBNB Price Prediction with Regression')

# Linear Regression
st.subheader('Linear Regression & Polynomial regression')

X = df['price'].values.reshape(-1,1)
y = df['reviews_per_month'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train,y_train)

y_lin_predict = lin_model.predict(X_test)



a = lin_model.coef_
b = lin_model.intercept_

poly_model = PolynomialFeatures(degree=3)
X_train_poly = poly_model.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_train_poly, y_train)
X_test_poly = poly_model.transform(X_test)
y_pred = pol_reg.predict(X_test_poly)

# Sort the values of X_test and the corresponding predictions
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test, y_pred), key=sort_axis)
X_test, y_poly_predict = zip(*sorted_zip)
# Plot
plt.title('Linear(blue) and Polynomial (red)')
plt.scatter(X, y, color='green')
plt.plot(X_train, a*X_train + b, color='blue')

plt.plot(X_test, y_poly_predict, color='red')
plt.xlabel('price')
plt.ylabel('reviews_per_month')
plt.show()
st.pyplot(plt)
plt.clf()


# Metrics
mae = metrics.mean_absolute_error(y_test, y_lin_predict)
mse = metrics.mean_squared_error(y_test, y_lin_predict)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_lin_predict))
st.write('Linear regression MAE,MSE, RMSE')
st.write(f'MAE: {mae}')
st.write(f'MSE: {mse}')
st.write(f'RMSE: {rmse}')

st.write('Linear regression Coef, inter, R2')
st.write(f'Coefficient: {lin_model.coef_}')
st.write(f'Intercept: {lin_model.intercept_}')
st.write(f'R2 Score: {lin_model.score(X,y)}')

st.write('Polynomial Regression')
mae = metrics.mean_absolute_error(y_test, y_poly_predict)
mse = metrics.mean_squared_error(y_test, y_poly_predict)
r2 = metrics.r2_score(y_test, y_poly_predict)

st.write(f'Mean Absolute Error: {mae}')
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')


#Multiple linear regression
st.subheader('Multiple Linear regression')

X = df[['neighbourhood_cleansed', 'beds', 'bathrooms','accommodates']]
y = df['price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi = LinearRegression()
multi.fit(X_train, y_train)

st.write(f'Intercept: {multi.intercept_}')
st.write(f'Coefficients: {multi.coef_}')

y_pred = multi.predict(X_test)

# Metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

st.write(f'Mean Absolute Error: {mae}')
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')

# Plot
plt.title('Multiple Linear Regression')
plt.scatter(X_test.index, y_test, color='blue', label='Actual')
plt.scatter(X_test.index, y_pred, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(plt)
plt.clf()
# Polynomial Regression
st.subheader('Polynomial Regression')
X = df[['availability_365','accommodates','neighbourhood_cleansed', 'beds','bathrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)



poly_model = PolynomialFeatures(degree=3)
X_train_poly = poly_model.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_train_poly, y_train)

X_test_poly = poly_model.transform(X_test)
y_poly_pred = pol_reg.predict(X_test_poly)


# Metrics
mae = metrics.mean_absolute_error(y_test, y_poly_pred)
mse = metrics.mean_squared_error(y_test, y_poly_pred)
r2 = metrics.r2_score(y_test, y_poly_pred)

st.write(f'Mean Absolute Error: {mae}')
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')




poly_five = PolynomialFeatures(degree=5)
X_poly = poly_five.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

columns = ['availability_365','accommodates','neighbourhood_cleansed', 'beds','bathrooms']

for i in range(X.shape[1]):
    y_pred = model.predict(X_poly)
    plt.scatter(X.iloc[:, i], y, color='blue', label='Original data')
    plt.scatter(X.iloc[:, i], y_pred, color='red', label='Poly prediction') 
    plt.legend(title=columns[i])
    plt.show()
    st.pyplot(plt)
    plt.clf()

from keras.models import load_model

model = load_model('ann_model.h5')
# Create inputs for the user to enter values
availability_365 = st.number_input('Enter how many days your property is available out of 365')
accommodates = st.number_input('Enter how many people the property can fit')
neighbourhood_cleansed = st.number_input('Enter neighborhood')
beds = st.number_input('how many beds/bedrooms?')
bathrooms = st.number_input('how many bathrooms?')
# Button to make prediction
if st.button('Predict Price'):
    # Create a numpy array of the inputs
    input_data = np.array([[availability_365, accommodates, neighbourhood_cleansed, beds, bathrooms]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'Predicted Price: {prediction[0][0]} for a 3-5 day stay')