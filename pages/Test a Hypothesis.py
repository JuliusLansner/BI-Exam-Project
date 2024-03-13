import os
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd
import scipy.stats as stats 
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2 
import matplotlib.pyplot as plt 

# paths
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cph_listings_df_clean.csv'))
file_pathI = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'media', 'hypo.png'))

# load data
cph_listings_df = pd.read_csv(file_path)

st.title('Explore your hypothesis')
st.subheader('Purpose and Description')
st.write('This page allows you to validate or reject different hypothesis based on categorical data.')

#image
logo = Image.open(file_pathI)
st.image(logo)

# create contingency table
contingency_table = pd.crosstab(cph_listings_df['host_is_superhost'], cph_listings_df['instant_bookable'])
contingency_table.index.name = None  # remove index name

# perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# interpret results
if p <= 0.05:
    result = 'Variables are dependent, reject Ho'
else:
    result = 'Variables are independent, Ho holds true'

# display results
st.subheader('Chi-square testing')
st.markdown('**Hypothesis:** Superhost and Instant Bookable are independent of each other')
st.write('Contingency Table:')
st.write(contingency_table)
st.write('Chi-square statistic:', chi2)
st.write('P-value:', p)
st.write('Degrees of freedom:', dof)
st.write('Expected frequencies:')
st.write(expected)
st.write('Interpretation:', result)

# plot chi-square distribution
alpha = 0.05
x = np.linspace(0, 30, 1000) 
y = stats.chi2.pdf(x, dof)
critical_value = stats.chi2.ppf(1 - alpha, dof)
fig, ax = plt.subplots()
ax.plot(x, y, label='Chi-Square Distribution (dof=1)')
ax.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5, label='Critical Region')
ax.axvline(chi2, color='blue', linestyle='dashed', label='Calculated Chi-Square')
ax.axvline(critical_value, color='green', linestyle='dashed', label='Critical Value')
ax.set_xlabel('Chi-Square Value')
ax.set_ylabel('Probability Density Function')
ax.set_title('Chi-Square Distribution and Critical Region')
ax.legend()
st.pyplot(fig)

st.title('Test your own categorical hypothesis')

# define the list of categorical features
categorical_features = ['instant_bookable', 'host_is_superhost', 'host_has_profile_pic',
                        'host_identity_verified', 'property_type',
                        'room_type', 'bathrooms']

# allow user to choose the features
selected_features = st.multiselect('Select two features for analysis:', categorical_features, [], key="feature_selection")

# check if exactly two features are selected
if len(selected_features) != 2:
    st.warning('Please select two features')
else:
    contingency_table = pd.crosstab(cph_listings_df[selected_features[0]], cph_listings_df[selected_features[1]])
    contingency_table.index.name = None 
    
    # perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # interpret results
    if p <= 0.05:
        result = 'Variables are dependent, reject Ho'
    else:
        result = 'Variables are independent, Ho holds true'

    # display results
    st.write('Contingency Table:')
    st.write(contingency_table)
    st.write('Chi-square statistic:', chi2)
    st.write('P-value:', p)
    st.write('Degrees of freedom:', dof)
    st.write('Expected frequencies:')
    st.write(expected)
    st.write('Interpretation:', result)

    # plot chi-square distribution
    alpha = 0.05
    x = np.linspace(0, 30, 1000)
    y = stats.chi2.pdf(x, dof)
    critical_value = stats.chi2.ppf(1 - alpha, dof)
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Chi-Square Distribution (dof=1)')
    ax.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5, label='Critical Region')
    ax.axvline(chi2, color='blue', linestyle='dashed', label='Calculated Chi-Square')
    ax.axvline(critical_value, color='green', linestyle='dashed', label='Critical Value')
    ax.set_xlabel('Chi-Square Value')
    ax.set_ylabel('Probability Density Function')
    ax.set_title('Chi-Square Distribution and Critical Region')
    ax.legend()
    st.pyplot(fig)