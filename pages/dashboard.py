import os
import pandas as pd
import textwrap
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# set the width of the entire app
st.set_page_config(layout="wide")

# path 
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cph_listings_df_clean.csv'))

# load data
cph_listings_df = pd.read_csv(file_path)

# center title
st.markdown("<h1 style='text-align: center;'>Visualising Airbnb Data</h1>", unsafe_allow_html=True)

# create 2 columns
col1, col2 = st.columns([1, 1])

neighbourhood_mapping = {1: 'Vesterbro-Kongens Enghave', 2: 'Nørrebro', 3: 'Indre By', 4: 'Østerbro',
                            5: 'Frederiksberg', 6: 'Amager Vest', 7: 'Amager st', 8: 'Bispebjerg',
                            9: 'Valby', 10: 'Vanløse', 11:'Brønshøj-Husum'}

cph_listings_df['neighbourhood_combined'] = cph_listings_df['neighbourhood_cleansed'].map(neighbourhood_mapping).astype(str)

# column 1
with col1:
    st.header("Hosts")

    # average acceptance superhosts
    avg_acceptance_rate_superhost = cph_listings_df[cph_listings_df['host_is_superhost'] == 1]['host_acceptance_rate'].mean()
    remaining_percentage_superhost = 100 - avg_acceptance_rate_superhost

    # average acceptance non-superhosts
    avg_acceptance_rate_non_superhost = cph_listings_df[cph_listings_df['host_is_superhost'] == 0]['host_acceptance_rate'].mean()
    remaining_percentage_non_superhost = 100 - avg_acceptance_rate_non_superhost
    
    # superhost donut chart 
    fig = px.pie(names=['Superhost', ''], values=[avg_acceptance_rate_superhost, remaining_percentage_superhost],
                 hole=0.6, color_discrete_sequence=['#636EFA', '#CCCCCC'])
    fig.update_layout(title='Avg. acc. rate for Superhosts', showlegend=False)

    # non-superhost donut chart
    fig2 = px.pie(names=['Non-Superhost', ''], values=[avg_acceptance_rate_non_superhost, remaining_percentage_non_superhost],
                 hole=0.6, color_discrete_sequence=['#EF553B', '#CCCCCC'])
    fig2.update_layout(title='Avg. acc. rate for Non-Superhosts', showlegend=False)

    # dist. of hosts
    superhost_counts = cph_listings_df['host_is_superhost'].value_counts()
    fig3 = px.pie(names=superhost_counts.index, values=superhost_counts.values,
                color_discrete_sequence=['#EF553B', '#636EFA'], title='Distribution of Superhosts vs. Non-Superhosts')
    fig3.update_layout(title_text='Dist. of Superhosts and Non-Superhosts', showlegend=False)

    # plot donut charts
    st.plotly_chart(fig.update_layout(height=400), use_container_width=True)
    st.plotly_chart(fig2.update_layout(height=400), use_container_width=True)
    st.plotly_chart(fig3.update_layout(height=400), use_container_width=True)

# column 2
with col2:
    st.header("Pricing")

    # bar chart avg price
    bar_chart = alt.Chart(cph_listings_df).mark_bar().encode(
    alt.Y('neighbourhood_cleansed:N', title='Neighbourhood'),
    alt.X('average(price):Q', title='Average Price'),
    color=alt.Color('neighbourhood_combined:N', title='Neighbourhood'),
    tooltip=['neighbourhood_combined:N', 'average(price):Q']
    ).properties(
        title='Average Neighbourhood Prices'
    )

    # horisontal bar chart 
    scatter_chart = alt.Chart(cph_listings_df).mark_circle().encode(
        alt.Y('neighbourhood_cleansed:N', title='Neighbourhood'),
        alt.X('price:Q', title='Price'),
        color=alt.Color('neighbourhood_combined:N', title='Neighbourhood'),
    ).properties(
        title='Distribution of Neighbourhood Prices'
    )

    # violin plot for price distribution
    fig, ax = plt.subplots()
    sns.violinplot(y=cph_listings_df['price'], palette="Set3", bw=0.5, ax=ax)
    ax.set_title('Price Distribution')

    st.altair_chart(bar_chart.properties(height=400), use_container_width=True)
    st.altair_chart(scatter_chart.properties(height=400), use_container_width=True)
    st.pyplot(fig, use_container_width=True)


st.header("Reviews")

# interactive scatter plot
scatter_plot = alt.Chart(cph_listings_df).mark_circle().encode(
    x=alt.X('review_scores_rating:Q', title='Review Score Rating'),
    y=alt.Y('price:Q', title='Price'),
    color=alt.Color('neighbourhood_combined:N', title='Neighbourhood'),
    tooltip=['price:Q', 'review_scores_rating:Q']
).properties(
    title='Price vs Review Score Rating'
).interactive()

st.altair_chart(scatter_plot, use_container_width=True)

st.header("Listings")

# interactive bar chart
bar_chart = alt.Chart(cph_listings_df).mark_bar().encode(
    x=alt.X('neighbourhood_cleansed:N', title='Neighbourhood'),
    y=alt.Y('count()', title='Number of Listings'),
    color=alt.Color('neighbourhood_combined:N', title='Neighbourhood'),
    tooltip=['neighbourhood_combined:N', 'count()']
).properties(
    title='Number of Listings per Neighbourhood'
).interactive()

st.altair_chart(bar_chart, use_container_width=True)
