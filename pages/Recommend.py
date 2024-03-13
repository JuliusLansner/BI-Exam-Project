import Recommend_function
from Recommend_function import get_recommendations
from Recommend_function import get_recommendations2
import streamlit as st

st.title('Rental Unit Recommendations')

# Define options for dropdowns
neighbourhood_options = ["Indre By", "Amager st", "Amager Vest","Vesterbro-Kongens Enghave",'Nørrebro','Frederiksberg','Vanløse','Østerbro','Valby','Bispebjerg','Brønshøj-Husum']

property_type_options = [
    'Private room in rental unit', 'Entire rental unit', 'Entire condo', 'Entire home', 
    'Private room in condo', 'Entire townhouse', 'Private room in townhouse', 
    'Room in aparthotel', 'Private room in home', 'Private room in villa', 
    'Entire villa', 'Entire loft', 'Private room in bed and breakfast', 
    'Entire guesthouse', 'Tiny home', 'Room in hotel', 'Entire serviced apartment', 
    'Private room in hostel', 'Private room in cabin', 'Houseboat', 
    'Shared room in hostel', 'Private room in guest suite', 'Entire cabin', 
    'Boat', 'Private room in vacation home', 'Private room in tiny home', 
    'Private room in loft', 'Private room in casa particular', 'Private room in boat', 
    'Entire guest suite', 'Private room in guesthouse', 'Private room in serviced apartment', 
    'Shared room in condo', 'Shared room in rental unit', 'Camper/RV', 'Entire bungalow', 
    'Entire vacation home', 'Entire place', 'Dome', 'Private room in castle', 
    'Private room in bungalow', 'Casa particular', 'Room in hostel', 'Barn', 
    'Room in boutique hotel', 'Farm stay', 'Shared room in bungalow', 'Hut', 
    'Shared room in bed and breakfast', 'Shared room in loft', 
    'Private room in shipping container', 'Entire cottage', 'Private room in hut', 
    'Private room', 'Shared room in hotel', 'Private room in barn'
]
room_type_options = ["Private room", "Entire home/apt", "Shared room",'Hotel room']

bathrooms_text_options = [
    '1 shared bath', '1 bath', '2 baths', '1.5 baths', '1 private bath', 
    '3 baths', 'Shared half-bath', '1.5 shared baths', '2 shared baths', 
    '2.5 baths', 'Half-bath', '0 shared baths', '0 baths', '3 shared baths', 
    '3.5 baths', '4.5 shared baths', '4 baths', '2.5 shared baths', '5 baths'
]

price_interval_options = ["100-200","200-500","500-800","800-1000","1000-2000","2000-10000","10000-100000"]

# Add dropdowns to the Streamlit app
neighbourhood = st.selectbox('Neighbourhood', neighbourhood_options)
property_type = st.selectbox('Property Type', property_type_options)
room_type = st.selectbox('Room Type', room_type_options)
bathrooms_text = st.selectbox('Bathrooms', bathrooms_text_options)
price_interval = st.selectbox('Price Interval', price_interval_options)


if st.button('Get Recommendations'):
    result = get_recommendations(neighbourhood, property_type, room_type, bathrooms_text, price_interval)

    st.write(result)

st.title('Write a message and see the recommendations')
user_input = st.text_input('Enter your preferences', 'Type here...')
if st.button('Get results'):
    recommendations2 = get_recommendations2(user_input)
    st.write(recommendations2)