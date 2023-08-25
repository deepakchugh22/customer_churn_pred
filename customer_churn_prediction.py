#!/usr/bin/env python
# coding: utf-8

# # Import dependencies
# 

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Load_dataset

# In[11]:


df = pd.read_excel("customer_churn_large_dataset.xlsx")
df.head()


# In[13]:


df.shape


# In[15]:


new_df = df.drop(columns=['CustomerID', 'Name'])
new_df.shape
new_df.head(5)


# # Perform some EDA to get to know your data better
# 
#      1: check for null values and duplicated columns
#      2: Perform Univariate Analysis (Check for distribution of data)
#      3: Perform Bivariate Analysis (Check for correlation b/w colns)
#  

# In[17]:


# check for null values
new_df.isna().sum()


# In[18]:


# check for duplicated values
new_df.duplicated().sum()


# In[19]:


# check brief description of data
new_df.describe()


# # checking  Distribution of Data

# In[20]:


# Loop through each column
for column in new_df.columns:
    # Determine the data type of the column
    dtype = new_df[column].dtype
    
    # If the column is numerical, create a histogram
    if dtype in ['int64', 'float64']:
        plt.figure(figsize=(5, 3))
        sns.histplot(new_df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    # If the column is categorical, create a count plot
    elif dtype == 'object':
        plt.figure(figsize=(5, 3))
        sns.countplot(data=new_df, x=column)
        plt.title(f'Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# # checking Correlation b/w columns

# In[21]:


new_df.sample(5)


# In[22]:


# checking for correlation b/w `Age ` and `Subscription_Length_Months` coln
sns.scatterplot(x=new_df['Subscription_Length_Months'], y=new_df['Age'], hue=new_df['Churn'])


# In[23]:


# checking for correlation b/w `Age ` and `Monthly_Bill` coln
sns.scatterplot(x=new_df['Monthly_Bill'], y=new_df['Age'], hue=new_df['Churn'])


# In[24]:


# checking for correlation b/w `Age ` and `Total_Usage_GB` coln
sns.scatterplot(x=new_df['Total_Usage_GB'], y=new_df['Age'], hue=new_df['Churn'])


# In[25]:


# checking for correlation b/w `Subscription_Length_Months` and `Total_Usage_GB` coln
sns.scatterplot(x=new_df['Subscription_Length_Months'], y=new_df['Total_Usage_GB'], hue=new_df['Churn'])


# # 
# By Analyzing this data at this extent, I have got to know this dataset is arranged in the manner as there is no relation b/w any columns and even distributions are very uniformly arranged. It suggests all columns holds equal importance in Model Building.
# 

# In[26]:


new_df.sample(5)


# #  Divide the data in `features` and `target`

# In[28]:


X = new_df.iloc[:, :-1]
y = new_df.iloc[:, -1]


# # Model building

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[31]:


# Differentiate `Categorical` & `Numerical` features
categorical_features = ['Gender', 'Location']
numerical_features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

# Create `StandardScalar` and `OneHotEncoder` Object
one_hot_encoder = OneHotEncoder()
standard_scalar = StandardScaler()


# Create ColumnTransformer Object for `Preprocessing Stuff`
preprocesser = ColumnTransformer(transformers=(
    ('encode_gender', one_hot_encoder, categorical_features),
    ('standardization', standard_scalar, numerical_features)
))


# In[32]:


# Create `Model Pipeline` for Logistic Regression
clf = Pipeline(steps=(
    ('preprocessing', preprocesser),
    ('classifier', LogisticRegression())
))


# In[33]:


clf.fit(X_train, y_train)
print("Accuracy score of Logistic Regression is: ", clf.score(X_test, y_test))


# In[34]:


# Check score using other metrics like `Precision Score`, `Recall Score`, `F1 Score`
y_pred = clf.predict(X_test)

print("The precision score of Logistic Regression is: ", precision_score(y_test, y_pred))
print("The recall score of Logistic Regression is: ", recall_score(y_test, y_pred))
print("The F1 score of Logistic Regression is: ", f1_score(y_test, y_pred))


# In[35]:


# Create `Model Pipeline` for `RandomForestClassifier` 
clf2 = Pipeline(steps=[
    ('preprocessing', preprocesser),
    ('classifier', RandomForestClassifier())
])


# In[36]:


clf2.fit(X_train, y_train)
print("The accuracy score of Random Forest Classifier is:", clf2.score(X_test, y_test))


# In[37]:


# Check score using other metrics like `Precision Score`, `Recall Score`, `F1 Score`
y_pred = clf2.predict(X_test)

print("The precision score of Logistic Regression is: ", precision_score(y_test, y_pred))
print("The recall score of Logistic Regression is: ", recall_score(y_test, y_pred))
print("The F1 score of Logistic Regression is: ", f1_score(y_test, y_pred))


# # 
# General Machine Learning is not showing that good of a result, but actually that's very logical as there was not any relation between features and Machine Learning tries to find the best line matching the relation. So, I hadn't expected anything more.

# # Now time to head towards Deep Learning, let's see how deep I can dig

# In[38]:


import tensorflow as tf
from tensorflow import keras


# In[39]:


# Create `Features` & `Targets`
features = preprocesser.fit_transform(X_train)
targets = y_train


# In[40]:


# Create Model using `Sequential` layer
model = keras.Sequential(layers=[
    keras.layers.Dense(units=64, activation="relu", input_shape=(features.shape[1], )),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=1, activation="sigmoid")
])


# In[41]:


# Add `Optimizer` and `Loss` function
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[42]:


# Check brief summary of model
model.summary()


# In[43]:


# Finally time to train the model
model.fit(x=features, y=targets, batch_size=1000, epochs=50, validation_split=0.2)


# # 
# Even a well curated Deep Learning model was not able to get accuracy over 51% on validation set. At the very least this was expected with the dataset it was training on.

# In[44]:


# Test the model
test_features = preprocesser.transform(X_test)
test_targets = y_test

model.evaluate(test_features, test_targets)


# # 
# Though I was not able to get better accuracy on this dataset, and I think dataset is also on partially fault there. But now let's move forward to next step and that is deploy this model on Streamlit Website.

# In[47]:


# Deploy the model using `pickle` module
import pickle

pickle.dump(clf, open("model.pkl", 'wb'))

