#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


pd.set_option('display.max_columns', None)
df = pd.read_csv("data/cleaned_data.csv")




df = df[['availability_365','reviews_per_month','neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'beds', 'price']]


from sklearn.cluster import KMeans
prices = df['price'].values.reshape(-1, 1)




num_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=9, value=5)
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(prices)

centers = kmeans.cluster_centers_

labels = kmeans.labels_

def cluster_of_price():
# Plot the data points and cluster centers
 plt.figure(figsize=(10, 6))
 plt.scatter(prices, np.zeros_like(prices), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
 plt.scatter(centers, np.zeros_like(centers), marker='x', c='r', s=100, label='Cluster Centers')
 plt.xlabel('Price')
 plt.title('KMeans Clustering of Prices')
 plt.legend()
 return plt

cluster_intervals = {}

# Loop over each cluster
for cluster_label in range(num_clusters):
    # Find the indices of data points belonging to the current cluster
    cluster_indices = np.where(labels == cluster_label)[0]
    
    # Extract the price values corresponding to the current cluster indices
    cluster_prices = prices[cluster_indices]
    
    # Compute the minimum and maximum price values for the current cluster
    min_price = np.min(cluster_prices)
    max_price = np.max(cluster_prices)
    
    # Store the interval for the current cluster
    cluster_intervals[cluster_label] = (min_price, max_price)




df['price_intervals'] = None

# Assign interval labels to each price based on the cluster
for cluster_label, interval in cluster_intervals.items():
    # Filter the DataFrame for the current cluster
    cluster_indices = np.where(labels == cluster_label)[0]
    df.loc[cluster_indices, 'price_intervals'] = f"{interval[0]}-{interval[1]}"




from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion="gini", 
    max_depth= 10,
    min_samples_split= 20,
    random_state=7)



from sklearn.model_selection import train_test_split

X = df.drop(columns=['price','price_intervals']).values
y = df['price_intervals'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)



random_forest_classifier.fit(X_train,y_train)


predictions = random_forest_classifier.predict(X_test)



from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)


def get_accuracy():
    return accuracy



from sklearn.model_selection import cross_val_score, KFold
n_folds = 6
cv_scores = cross_val_score(random_forest_classifier, X, y, cv=KFold(n_folds, shuffle=True, random_state=42))

def get_cross_val_score():
    return cv_scores



from sklearn.metrics import confusion_matrix, classification_report
conf_matrix = confusion_matrix(y_test, predictions)






df_features = df.drop(columns=['price'])



feature = df_features.columns
importances = random_forest_classifier.feature_importances_
indices = np.argsort(importances)

def feature_importances():
 plt.clf()
 plt.title('Feature importances')      
 plt.barh(range(len(indices)), importances[indices], color='b', align='center')
 plt.yticks(range(len(indices)), [feature[i] for i in indices])
 plt.xlabel('Relative Importance')
 return plt        

def predict_rental(availability_365,reviews_per_month,neighbourhood, property_type, room_type, accommodates, bathrooms, beds):
    neighbourhood_map = {
        'Vesterbro-Kongens Enghave': 1, 'Nørrebro': 2, 'Indre By': 3, 'Østerbro': 4,
        'Frederiksberg': 5, 'Amager Vest': 6, 'Amager st': 7, 'Bispebjerg': 8,
        'Valby': 9, 'Vanløse': 10, 'Brønshøj-Husum': 11
    }
        
    room_type_map = {
        'Entire home/apt': 1, 'Private room': 2, 'Shared room': 3, 'Hotel room': 4
    }
    
    property_mapping = {
        'Entire rental unit': 1,
        'Entire condo': 2,
        'Private room in rental unit': 3,
        'Entire home': 4
    }
    
    neighbourhood_numeric = neighbourhood_map.get(neighbourhood, 0)
    room_type_numeric = room_type_map.get(room_type, 0)
    property_type_numeric = property_mapping.get(property_type, 0)
    
    input_x = [[availability_365,reviews_per_month,neighbourhood_numeric, property_type_numeric, room_type_numeric, accommodates, bathrooms, beds]]
    prediction = random_forest_classifier.predict(input_x)
    return prediction




#Showing improvement methods for price prediction model
st.title('Improvememt of model with random forest and clustering')

plt = cluster_of_price()
st.pyplot(plt)

#Show price intervals
unique_intervals = df['price_intervals'].unique()
st.write("Price Intervals:")
for interval in unique_intervals:
    st.write(interval)

st.header("Random Forest Classifier Parameters:")
st.text("- n_estimators: 100")
st.text("- criterion: gini")
st.text("- max_depth: 10")
st.text("- min_samples_split: 20")
st.text("- random_state: 7")

st.header('Model Evaluation')
st.write("Accuracy of the model is: ", get_accuracy())

cv_scores = get_cross_val_score()
st.write(f'Cross-Validation Scores: {cv_scores}')

#Show feature importances
plt = feature_importances()
st.pyplot(plt)

st.header('Fill out the form and predict Rental Price interval')
#Setup inputs for prediction function
reviews_per_month = st.number_input('Reviews per month', value=0.0)
neighbourhood = st.selectbox('Neighbourhood', ['Vesterbro-Kongens Enghave', 'Nørrebro', 'Indre By', 'Østerbro',
                                               'Frederiksberg', 'Amager Vest', 'Amager st', 'Bispebjerg',
                                               'Valby', 'Vanløse', 'Brønshøj-Husum'])
property_type = st.selectbox('Property Type', ['Entire rental unit', 'Entire condo',
                                               'Private room in rental unit', 'Entire home'])
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
accommodates = st.number_input('Accommodates', value=1)
bathrooms = st.number_input('Bathrooms', value=1.0)
beds = st.number_input('Beds', value=1.0)
availability_365 = st.number_input('Availability 365', value=0)

#Predict function
if st.button('Predict Rental Price interval'):
    prediction = predict_rental(availability_365, reviews_per_month, neighbourhood, property_type, room_type, accommodates, bathrooms, beds)
    st.write(f'Predicted rental price interval: {prediction}')


