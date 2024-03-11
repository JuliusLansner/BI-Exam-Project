import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
file_path = "/Users/mikkel/Documents/GitHub/cph_airbnb/data/cph_listings_df_clean.csv"
cph_listings_df = pd.read_csv(file_path)


# Add a title
st.title("Analyse af Airbnb Data")


import streamlit as st

# Opret tre kolonner
col1, col2, col3 = st.columns(3)

# Kolonne 1
with col1:
    st.header("Kolonne 1")
    # Tilføj dit indhold til kolonne 1

# Kolonne 2
with col2:
    st.header("Kolonne 2")
    # Tilføj dit indhold til kolonne 2

# Kolonne 3
with col3:
    st.header("Kolonne 3")

    neighbourhood_mapping = {'Vesterbro-Kongens Enghave': 1, 'Nørrebro': 2, 'Indre By': 3, 'Østerbro': 4,
                         'Frederiksberg': 5, 'Amager Vest': 6, 'Amager st': 7, 'Bispebjerg': 8,
                         'Valby': 9, 'Vanløse': 10, 'Brønshøj-Husum': 11}
    st.header("Average price pr. neighbourhood")
    
    # Grouper data efter kvarter og gennemsnitlig pris
    neighbourhood_prices = cph_listings_df.groupby('neighbourhood_cleansed')['price'].mean()

    # Opret en ny kolonne med gennemsnitlig pris pr. kvarter
    cph_listings_df['price_per_neighbourhood'] = cph_listings_df['neighbourhood_cleansed'].map(neighbourhood_prices)

    # Barplot for gennemsnitlig pris pr. kvarter
    plt.figure(figsize=(10, 6))
    sns.barplot(x=neighbourhood_prices.index, y=neighbourhood_prices.values, palette="viridis")

    plt.title('Gennemsnitlig pris pr. kvarter')
    plt.xlabel('Kvarter')
    plt.ylabel('Gennemsnitlig pris')

    # Tilføj forklarende overskrifter uden for plottet
    plt.legend(labels=[f"{num}: {neighbourhood_mapping[num]}" for num in neighbourhood_mapping], title='Kvarter', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Vis plottet
    st.pyplot(plt)


# Question 1: Hvad påvirker prisen på ens Airbnb
st.header("1. Hvad påvirker prisen på ens Airbnb?")


# groupe data by neightbourhood and average price
neighbourhood_prices = cph_listings_df.groupby('neighbourhood_cleansed')['price'].mean()

# create a new column with average price per neighbourhood
cph_listings_df['price_per_neighbourhood'] = cph_listings_df['neighbourhood_cleansed'].map(neighbourhood_prices)

# Question 2: Kan man øge indkomsten fra ens Airbnb
st.header("2. Kan man øge indkomsten fra ens Airbnb?")

# Question 3: Hvilke Airbnb typer tjener mest per år
st.header("3. Hvilke Airbnb typer tjener mest per år?")




# Question 4: Kan anmeldelser påvirke ens salg
st.header("4. Kan anmeldelser påvirke ens salg?")





# Feature engineering
feature_data = cph_listings_df[['price', 'number_of_reviews']]
scaler = StandardScaler()
standardized_data = scaler.fit_transform(feature_data)

# K-Means clustering
num_clusters = range(1, 11)
inertia_values = []

for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(standardized_data)
    inertia_values.append(kmeans.inertia_)

# Set number of clusters
num_clusters = 9

# Create KMeans model
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
kmeans.fit(standardized_data)

# Predict clusters
cluster_predictions = kmeans.predict(standardized_data)

# Create KMeans model
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
kmeans.fit(standardized_data)

# Explore centroids
cluster_centers = kmeans.cluster_centers_

# Predict clusters
cluster_predictions = kmeans.predict(standardized_data)



# 3D Scatter Plot
st.header("3D Scatter Plot af KMeans Clusters")

df_3d_cluster = pd.DataFrame({
    'feature1': standardized_data[:, 0],
    'feature2': standardized_data[:, 1],
    'cluster_label': cluster_predictions
})

fig = px.scatter_3d(df_3d_cluster, x='feature1', y='feature2', z='cluster_label', color='cluster_label',
                    opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set1)

fig.update_layout(scene=dict(xaxis_title='Price', yaxis_title='Number of reviews', zaxis_title='Cluster'),
                  coloraxis_colorbar=dict(title='Cluster'))

st.plotly_chart(fig)



