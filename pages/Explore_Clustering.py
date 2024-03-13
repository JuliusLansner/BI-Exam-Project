import os
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import SilhouetteVisualizer

# paths
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cph_listings_df_clean.csv'))
model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(model_folder, 'kmeans_model.joblib')

# load data
cph_listings_df = pd.read_csv(file_path)

# load model 
kmeans_model = joblib.load(model_path)

st.title('Copenhagen Airbnb Clustering')
st.subheader('Purpose and Description')
st.write('This page aims to analyze Airbnb listings in Copenhagen through clustering analysis.')
st.write('The purpose of this analysis is to gain insights into the pricing and review patterns of Airbnb listings.')
st.write('Users can explore different clusters and get insight into the characteristics of each cluster.')

# data preparation
scaler = StandardScaler()
feature_data = cph_listings_df[['price', 'number_of_reviews']]
standardized_data = scaler.fit_transform(feature_data)

# method to perform KMeans clustering
def perform_kmeans(data):
    cluster_predictions = kmeans_model.predict(data)
    silhouette_avg = silhouette_score(data, cluster_predictions)
    return cluster_predictions, silhouette_avg

# input for number of clusters
st.sidebar.title('KMeans Clustering')
st.sidebar.subheader('Here you can adjust the number of clusters')
num_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=9, value=9)

# perform KMeans clustering
cluster_predictions, silhouette_avg = perform_kmeans(standardized_data)

st.subheader('Clusters visualized')

# visualize clusters
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=standardized_data[:, 0], y=standardized_data[:, 1], hue=cluster_predictions, palette='pastel', s=100, alpha=0.7, ax=ax)
ax.set_title('KMeans Clusters')
ax.set_xlabel('Price')
ax.set_ylabel('Number of reviews')
ax.legend(title='Clusters')

# add labels centered on cluster means
for label in range(num_clusters):
    cluster_mean = standardized_data[cluster_predictions == label].mean(axis=0)
    ax.text(cluster_mean[0], cluster_mean[1], str(label), 
             size=15, weight='bold', color='white', backgroundcolor=sns.color_palette('pastel')[label],
             horizontalalignment='center', verticalalignment='center')

st.pyplot(fig)

st.subheader('View the silhouette score')
# display silhouette score
st.write(f'Silhouette Score: {silhouette_avg:.3f}')

st.subheader('Get a better insight and feeling of each cluster')
# 3D scatter plot
df_3d_cluster = pd.DataFrame({
    'feature1': standardized_data[:, 0],
    'feature2': standardized_data[:, 1],
    'cluster_label': cluster_predictions
})

fig_3d = px.scatter_3d(df_3d_cluster, x='feature1', y='feature2', z='cluster_label',
                        color='cluster_label', opacity=0.7, color_discrete_sequence=px.colors.qualitative.Dark24)
fig_3d.update_layout(scene=dict(xaxis_title='Price', yaxis_title='Number of reviews', zaxis_title='Cluster'))
st.plotly_chart(fig_3d)

st.subheader('Input own numbers and predict the cluster')
# user input for price and number of reviews
price = st.number_input('Enter price:')
num_reviews = st.number_input('Enter number of reviews:')

# scale the user input
scaled_input = scaler.transform([[price, num_reviews]])

# predict the cluster for user input
predicted_cluster = kmeans_model.predict(scaled_input)

st.write(f'Your input belongs to cluster {predicted_cluster[0]}')

# explore centroids
cluster_centers = kmeans_model.cluster_centers_

# invert standardization process to get original values
original_centers = scaler.inverse_transform(cluster_centers)

# create DataFrame to display original centroids
original_centers_df = pd.DataFrame(original_centers, columns=['Original Price', 'Original Number of Reviews'])

st.subheader('View centroids')
# display original centroids
st.write("Original Centroids:")
st.write(original_centers_df)

# sidebar input for cluster selection
st.sidebar.title('Select a  cluster that fits your needs')
cluster_names = {
    0: 'Lower End',
    1: 'Mid',
    2: 'Tight budget',
    3: 'Cheap and Reviewed',
    4: 'Lower End Reviewed',
    5: 'High End',
    6: 'Lower End',
    7: 'Low - Mid Tested',
    8: 'Lower Mid ',
}

selected_cluster_name = st.sidebar.selectbox('Select cluster:', options=list(cluster_names.values()))

# get the cluster number based on selected name
selected_cluster_num = [key for key, value in cluster_names.items() if value == selected_cluster_name][0]

# display selected cluster name
st.write(f'Selected Cluster: {selected_cluster_num} - {selected_cluster_name}')
st.write("Choose a cluster in the sidebar that you find fitting, and see the listings that belong to that cluster")

# filter listings based on selected cluster
filtered_listings = cph_listings_df[cluster_predictions == selected_cluster_num]

# add cluster column to filtered listings
filtered_listings['Cluster'] = selected_cluster_num

# display filtered listings
st.write('Filtered Listings:')
st.write(filtered_listings)