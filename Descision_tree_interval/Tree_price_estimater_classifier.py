#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv("cleaned_data.csv")
pd.set_option('display.max_columns', None)


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.dtypes


# Dropping non numeric columns

# In[6]:


df.drop(columns=['availability_interval','acceptance_rate_interval'],inplace=True)


# In[7]:


df = df[['availability_365','reviews_per_month','neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'beds', 'price']]

def drop_high_prices(price_cut):
    df.drop(df[df['price'] > price_cut].index, inplace=True)

def drop_high_accommodates(accommodates_cut):
    df.drop(df[df['accommodates'] > accommodates_cut].index, inplace=True)

# In[8]:


num_bins = [0,100,200,500,800,1000,2000,10000,100000]

df['price_intervals'] = pd.cut(df['price'],bins=num_bins) 


# In[9]:


df['price_intervals'] = df['price_intervals'].astype(str)


# In[10]:


df


# In[11]:


df.shape


# Starting training of descision tree model

# In[12]:


from sklearn import tree
params = {'max_depth': 5}
model = tree.DecisionTreeClassifier(**params)


# Splitting values from df into x and y and making training and test sets.

# In[13]:


from sklearn.model_selection import train_test_split

X = df.drop(columns=['price','price_intervals']).values
y = df['price_intervals'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# Training model.

# In[14]:


model.fit(X_train, y_train)


# In[18]:


import graphviz
def getTree():
 gr_data = tree.export_graphviz(model, out_file=None, 
                         feature_names=df.columns[:8], class_names = True,        
                         filled=True, rounded=True, proportion = False, special_characters=True)  
 dtree = graphviz.Source(gr_data)
 return dtree

# In[19]:



# In[20]:



# See the classes.

# In[21]:


classes = model.classes_

for i, class_label in enumerate(classes):
    print(f"Class {i+1}: {class_label}")


# # Predict values

# Predict values and show predictions.

# In[22]:


predictions = model.predict(X_test)


# In[23]:


predictions2 = model.predict(X)


# In[24]:


for index, prediction in enumerate(predictions):
    print(f"Index {index}: Predicted class {prediction}")


# # See how good the model is.

# model 1

# In[25]:


from sklearn.metrics import accuracy_score

def accuracy_test_set():
 accuracy = accuracy_score(y_test, predictions)
 accuracy
 return accuracy


# Doing another test on the whole dataset.

# In[26]:

def accuracy_whole_set():
 accuracy2 = accuracy_score(y,predictions2)
 accuracy2
 return accuracy2


# Doing cross_val_scores.

# In[27]:


from sklearn.model_selection import cross_val_score, KFold

def cross_val_scores():
 n_folds = 5
 cv_scores = cross_val_score(model, X, y, cv=KFold(n_folds, shuffle=True, random_state=42))
 return cv_scores



# Checking feature importances.

# In[28]:


model.feature_importances_


# In[29]:


df_features = df.drop(columns=['price'])
df_features


# In[30]:


feature = df_features.columns
importances = model.feature_importances_
indices = np.argsort(importances)

def feature_imprtances():
 plt.clf()
 plt.title('Feature importances')      
 plt.barh(range(len(indices)), importances[indices], color='b', align='center')
 plt.yticks(range(len(indices)), [feature[i] for i in indices])
 plt.xlabel('Relative Importance')
 plt.show() 
 return plt

# model 2

# Making a function to predict price interval.

# In[31]:


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
    prediction = model.predict(input_x)
    return prediction


# In[35]:
def availability_prices():
 plt.clf()
 bin_edges = [0, 50, 100, 150, 200, 250, 300, 365]

 bin_labels = ['0-49', '50-99', '100-149', '150-199', '200-249', '250-299', '300-365']

 df['availability_interval'] = pd.cut(df['availability_365'], bins=bin_edges, labels=bin_labels, include_lowest=True)

 availability_price_means = df.groupby('availability_interval')['price'].mean()

 availability_interval_counts = df['availability_interval'].value_counts()

 fig, ax1 = plt.subplots(figsize=(10, 6))

 color = 'tab:blue'
 ax1.set_xlabel('Availability Interval (in days)')
 ax1.set_ylabel('Count', color=color)
 ax1.bar(availability_interval_counts.index, availability_interval_counts, color=color)
 ax1.tick_params(axis='y', labelcolor=color)
 ax1.grid(axis='y', linestyle='--', alpha=0.7)

 ax2 = ax1.twinx()

 color = 'tab:red'
 ax2.set_ylabel('Average Price', color=color)
 ax2.plot(availability_price_means.index, availability_price_means, color=color, marker='o', linestyle='-')
 ax2.tick_params(axis='y', labelcolor=color)
 plt.title('Availability Distribution and Average Price by Intervals')

 plt.xticks(rotation=45)

 plt.tight_layout()
 plt.show()
 return plt

# In[36]:


df['accommodates'].unique()


# In[42]:

def accommodates_prices():
 plt.clf()
 accommodates_summary = df.groupby('accommodates')['price'].agg(['count', 'mean'])

 fig, ax1 = plt.subplots(figsize=(10, 6))

 color = 'skyblue'
 ax1.set_xlabel('Accommodates')
 ax1.set_ylabel('Count', color=color)
 ax1.bar(accommodates_summary.index, accommodates_summary['count'], color=color)
 ax1.tick_params(axis='y', labelcolor=color)

 ax2 = ax1.twinx()
 color = 'orange'
 ax2.set_ylabel('Average Price ($)', color=color)
 ax2.plot(accommodates_summary.index, accommodates_summary['mean'], color=color, marker='o')
 ax2.tick_params(axis='y', labelcolor=color)

 plt.title('Accommodates Counts and Average Prices')
 plt.grid(axis='y', linestyle='--', alpha=0.7)
 plt.xticks(rotation=45)  
 plt.tight_layout()
 plt.show()
 return plt

def outliers_price():
 plt.clf()
 Q1 = df['price'].quantile(0.25)
 Q3 = df['price'].quantile(0.75)

 IQR = Q3 - Q1

 lower_bound = Q1 - 1.5 * IQR
 upper_bound = Q3 + 1.5 * IQR

 outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

 plt.figure(figsize=(8, 6))
 plt.boxplot(df['price'], vert=False)
 plt.scatter(outliers['price'], [1] * len(outliers), color='red', label='Outliers')
 plt.title('Boxplot of Price with Outliers')
 plt.xlabel('Price')
 plt.yticks([])
 plt.legend()
 plt.show()
 return plt

def outliers_accommodates():
    plt.clf()
    Q1 = df['accommodates'].quantile(0.25)
    Q3 = df['accommodates'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['accommodates'] < lower_bound) | (df['accommodates'] > upper_bound)]

    plt.figure(figsize=(8, 6))
    plt.boxplot(df['accommodates'], vert=False)
    plt.scatter(outliers['accommodates'], [1] * len(outliers), color='red', label='Outliers')
    plt.title('Boxplot of Accommodates with Outliers')
    plt.xlabel('Accommodates')
    plt.yticks([])
    plt.legend()
    plt.show()
    return plt
# In[ ]:



