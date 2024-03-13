import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# title 
st.markdown("<h1 style='text-align: center;'>Airbnbs in Copenhagen 2023</h1>", unsafe_allow_html=True)

#image
logo = Image.open('media/airbnb.png')
st.image(logo, use_column_width=True)

# intro
st.write("In a world where travel is more prevalent than ever, and the competition for accommodations is increasing, aswell as the search for unique experiences, Airbnb remains more relevant than ever.")
