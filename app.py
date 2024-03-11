import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# streamlit run app.py

# title 
st.markdown("<h1 style='text-align: center;'>Airbnbs in Copenhagen 2023</h1>", unsafe_allow_html=True)

# # image
logo = Image.open('media/airbnb.png')
st.image(logo, use_column_width=True)

# intro
st.write("In a world were travel is more prevalent than ever, and with the increasing competition for accommodations, aswell as a search for unique experiences, Airbnb remains more relevant than ever.")
st.write("Therefore we deciced to explore the Airbnb market in Copenhagen, to see if we could find some interesting insights.")
st.write("Here are the questions we sought to answer:")

# problem statement? maybe your predictions? 
st.write("- What affects the price of an Airbnb?")
st.write("- Can you increase your revenue from your Airbnb?")
st.write("- Does review affect your Airbnb?")






st.write("Hvilke Airbnb typer tjener mest per år.")
st.write("Påvirker befolkningstætheden prisen på Airbnb's?")
st.write("Påvirker befolkningstætheden antallet af Airbnb's?")
st.write("Er priserne højere der, hvor der er høj befolkningstæthed?")

