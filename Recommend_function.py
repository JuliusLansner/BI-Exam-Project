#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


na_values = ['NaN','Na','N/A']
df = pd.read_csv('cleaned_data_before_numeric_change.csv',na_values=na_values)


# In[3]:


df


# Feature engineering price intervals.

# In[4]:


num_bins = [0,100,200,500,800,1000,2000,10000,100000]

df['price_intervals'] = pd.cut(df['price'],bins=num_bins)


# In[5]:


df.isnull().sum()


# Taking parameters i wish to check for which is property_type, room_type, bathrooms_text, beds, price

# In[6]:


selected_columns = ['neighbourhood_cleansed','property_type', 'room_type', 'bathrooms_text','price_intervals']

selection_df = df[selected_columns].copy()


# In[7]:


selection_df


# Taking vectorizer from sklearn. Using it do split data into documents and calculate TF and IDF and make vectors from each documents.

# In[8]:


documents = selection_df['neighbourhood_cleansed'] + " " + selection_df['property_type'] + ' ' + selection_df['room_type'] + ' ' + selection_df['bathrooms_text'] + ' ' + selection_df['price_intervals'].astype(str)
documents


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)


# In[10]:


tfidf_matrix.shape


# In[11]:


#For computing cosine between vectors
from sklearn.metrics.pairwise import linear_kernel


# In[12]:


def get_recommendations(neighbourhood, property_type, room_type, bathrooms_text, price_interval):
    inputt = neighbourhood + " " + property_type + " " + room_type + " " + bathrooms_text + " " + price_interval
    
    input_vector = tfidf.transform([inputt])

    similarity_scores = linear_kernel(input_vector, tfidf_matrix)

    top_indices = similarity_scores.argsort()[0][::-1][:10]

    top_similar_rental_units = df.iloc[top_indices]
    
    return top_similar_rental_units


# In[13]:


def get_recommendations2(user_input):
    
    input_vector = tfidf.transform([user_input])

    similarity_scores = linear_kernel(input_vector, tfidf_matrix)

    top_indices = similarity_scores.argsort()[0][::-1][:10]

    top_similar_rental_units = df.iloc[top_indices]
    
    return top_similar_rental_units


# In[14]:


pd.set_option("display.max_columns", 40)


# In[15]:


get_recommendations('indre','Private room in rental unit','Private','1','500')


# In[16]:


get_recommendations('indre','Private','1','shared','1000')


# In[17]:


get_recommendations('nørrebro','Private','1','shared','200')


# In[18]:


get_recommendations2("indre by price 1000")

