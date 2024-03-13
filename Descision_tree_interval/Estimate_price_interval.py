import Tree_price_estimater_classifier
from Tree_price_estimater_classifier import predict_rental, accuracy_test_set, accuracy_whole_set, cross_val_scores, feature_imprtances, availability_prices, accommodates_prices, outliers_price, drop_high_prices, outliers_accommodates, drop_high_accommodates
import graphviz
import streamlit as st

# Clear cache for functions that modify the DataFrame
drop_high_prices = st.cache_data(drop_high_prices)
drop_high_accommodates = st.cache_data(drop_high_accommodates)

# Viser titlen på appen
st.title('Model to predict rental price interval')

# Viser model og input til den
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


if st.button('Predict Rental Price interval'):
    prediction = predict_rental(availability_365, reviews_per_month, neighbourhood, property_type, room_type, accommodates, bathrooms, beds)
    st.write(f'Predicted rental price interval: {prediction}')

# Viser test resultater
test_score = accuracy_test_set() 
st.write(f'Test score: {test_score}')  

test_score_whole = accuracy_whole_set()
st.write(f'Whole set score: {test_score_whole}')

cv_scores = cross_val_scores()
st.write(f'Cross-Validation Scores: {cv_scores}')

st.header('Cut some of the outliers -> see how it affects the model accuracy.')
cut_price = st.slider('Cut Price:', min_value=0, max_value=10000, value=12000, step=100)

cut_accommodates = st.slider('Cut accommodates:', min_value=0, max_value=16, value=16, step=1)

if st.button('Drop High Prices'):
    drop_high_prices(cut_price)
    

if st.button('Drop High Accommodates'):
    drop_high_accommodates(cut_accommodates)
    

plt = outliers_price()
st.pyplot(plt)

plt = outliers_accommodates()
st.pyplot(plt)

# Viser diverse grafer.
plt = feature_imprtances()
st.pyplot(plt)

plt = availability_prices()
st.pyplot(plt)

plt = accommodates_prices()
st.pyplot(plt)
